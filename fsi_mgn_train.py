"""Training of FSIGraphNet (fsi_mgn): physics-constrained, multi-step, autonomous.

Every window: known CFD frames up to the anchor -> burn-in (the same model rolls out on its own outputs, no gradient)
-> H autonomous steps of model.rollout (one forward per step: fixed coupling stages with explicit physics feedback)
-> labels read only now -> loss on every coupling stage of every step -> backward -> AdamW.
Nothing is solved or iterated per time step; the only iterations are optimizer updates and multiplier updates.

Loss per stage s (weight stage_weights[s]):
  data        cell U, one-step dU, grid-scale dU, p, face flux, vorticity; y, v, a; wall loads Fy/Fp/Fv, near-wall p;
              one-step increments of p, near-wall p and Fy
  constraints augmented Lagrangian on h_j/tol_j <= 1:  lambda_j h_j/tol_j + rho_j/2 (h_j/tol_j)^2   (j = fsi, cont, wall)
                h_fsi = |m a + c v + k y - Fy(U, p)| / sF,  h_cont = rms(sum_f phi / cell capacity),
                h_wall = rms((phi - mesh flux) / (U_inf |S|)) on the rear wall
              Newmark kinematic consistency of (y, v, a); stage decrease relu(h_j(s) - h_j(s-1))
  bias        window-mean error of lift and displacement (final stage), squared
  oscillation window std of the lift and window-mean lift power Fy*v vs CFD (final stage), squared
Multipliers: every `every` updates, lambda_j <- max(0, lambda_j + rho_j (mean h_j / tol_j - 1)).

Data parallel with torchrun (gloo; gradients averaged on the host in FP64).
Container mounts: /work (code), /contracts/config.json, /views (TRAIN cases), /results, /control, /init.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, os, random, sys, time, traceback


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def mem_gib():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) / 1048576


def lr_factor(u, warmup, total, floor):
    if u <= warmup: return u / max(1, warmup)
    t = min(1.0, (u - warmup) / max(1, total - warmup)); return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))


def horizon_at(schedule, u):
    return [h for s, h in schedule if u >= s][-1]


CONSTRAINTS = ('fsi', 'cont', 'wall')


class Labels:
    """CFD labels of ONE TRAIN case (read only after a rollout) + TRAIN-only per-case fluctuation scales."""
    def __init__(self, c):
        import numpy as np
        z = c.z; self.U = z('native_cell_U'); self.p = z('native_cell_p'); self.phi = z('absolute_flux')
        self.y = np.array([f['centreOfRotation'][1] for f in c.frames]); self.v = np.array([f['velocity'][1] for f in c.frames])
        self.a = np.array([f['acceleration'][1] for f in c.frames])
        U = np.array(self.U)[..., :2]; p = np.array(self.p); S = np.array(z('face_sf'))[..., :2]; A = np.linalg.norm(S, axis=-1)
        un = np.array(self.phi) / A; F = np.array(z('native_force_components')); self.F = F   # [Fy, Fp, Fv] per frame
        self.scales = {'u': float(np.sqrt(((U - U.mean(0)) ** 2).sum(-1).mean())), 'p': float(np.sqrt(((p - p.mean(0)) ** 2).mean())),
                       'phi': float(np.sqrt(((un - un.mean(0)) ** 2).mean())), 'du': float(np.sqrt((np.diff(U, axis=0) ** 2).sum(-1).mean())),
                       'y': float(self.y.std()), 'v': float(self.v.std()), 'a': float(self.a.std()), 'Fy': float(F[:, 0].std()),
                       'Fp': float(F[:, 1].std()), 'Fv': float(F[:, 2].std()),
                       'dp': float(np.sqrt((np.diff(p, axis=0) ** 2).mean())), 'dFy': float(np.sqrt((np.diff(F[:, 0]) ** 2).mean()))}
        own = np.array(z('planar_face_owner')); nei = np.array(z('planar_face_neighbor')); inner = nei >= 0; nei_ = np.where(inner, nei, own)
        vol = np.array(z('cell_volume')); ws = []
        for k in range(0, len(U), 5):
            Uf = np.where(inner[:, None], .5 * (U[k][own] + U[k][nei_]), U[k][own]); cz = S[k][:, 0] * Uf[:, 1] - S[k][:, 1] * Uf[:, 0]
            w = np.zeros(len(vol[k])); np.add.at(w, own, cz); np.add.at(w, nei[inner], -cz[inner]); ws.append(w / vol[k])
        ws = np.array(ws); self.scales['w'] = float(np.sqrt(((ws - ws.mean(0)) ** 2).mean()))


def main():
    import numpy as np, torch, torch.distributed as dist, faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    dist.init_process_group('gloo'); rank, world = dist.get_rank(), dist.get_world_size()
    cfg = json.loads(Path('/contracts/config.json').read_text()); out = Path('/results'); mc = cfg['model']
    manifest = json.loads(Path('/work/SOURCE_MANIFEST.json').read_text())
    if any(sha(Path('/work') / k) != v for k, v in manifest.items()): raise ValueError('Source pins')
    from fsi_mgn.data import Case, Normalizer, compute_normalizer
    from fsi_mgn.model import FSIGraphModel
    from fsi_mgn import physics as ph
    if cfg.get('gpu_memory_fraction'): torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8': raise ValueError('Deterministic runtime required (CUBLAS_WORKSPACE_CONFIG)')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    seed = int(cfg['seed']) + rank; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed); torch.set_num_threads(2)
    dev = 'cuda'; NK = int(mc['known_frames'])
    Hmax = max(h for _, h in cfg['horizon_schedule']) + max(max(ch) for _, ch in cfg['burn_in_schedule'])
    cases = [Case(Path('/views') / c['name'], c['sequence_sha256'], c['report_sha256'], dev) for c in cfg['train_cases']]
    labels = [Labels(c) for c in cases]
    W = cfg['loss_weights']; SW = cfg['stage_weights']; AL = cfg['augmented_lagrangian']
    lam = {j: 0.0 for j in CONSTRAINTS}
    # burn-in schedule [[from_update, [choices]], ...]: from scratch, start with 0 until one-step prediction is learned
    burn_choices_at = lambda u: [ch for s0, ch in cfg['burn_in_schedule'] if u >= s0][-1]
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'world': world, 'config': cfg} if rank == 0 else None

    def phase_rate_err(yp, vp, yc, vc, w0):
        """Squared relative error of the summed per-step phase advance of the displacement phasor (y, -v/w0)."""
        adv = lambda th: torch.remainder(th[1:] - th[:-1] + math.pi, 2 * math.pi) - math.pi
        dp = adv(torch.atan2(-vp / w0, yp)).sum(); dc = adv(torch.atan2(-yp.new_tensor(vc) / w0, yp.new_tensor(yc))).sum()
        return ((dp - dc) / dc.abs().clamp_min(1e-6)).square()

    def window_loss(ci, off, outs, burn, start_U):
        """Loss on every coupling stage (1..K) of every step; returns total, scalar terms, constraint values per stage."""
        c, L = cases[ci], labels[ci]; s = L.scales; a = off + NK - 1 + burn; acc = {}; hs = {}
        wall_cells = torch.unique(c.owner[c.front | c.rear])                      # cells touching either cylinder
        T = lambda x: torch.as_tensor(np.array(x), dtype=torch.float64, device=dev)
        Uprev = start_U; total = 0.0; n = len(outs); eF, ey, Fp_, Fc_, Pp_, Pc_, Qp_, Qc_ = [], [], [], [], [], [], [], []; y0 = float(L.y.mean())
        yp_, vp_, yc_, vc_ = [], [], [], []
        pprev = outs[0][1].stages[0].p; Fprev = outs[0][1].stages[0].ev.Fy   # stage 0 = the window's start flow (no parameters)
        for i, (st, out) in enumerate(outs):
            k = a + 1 + i
            if abs(st.time_s - float(c.times[k])) > 1e-8: raise ValueError('Label clock mismatch')
            Uc, pc, phc, Uc0 = T(L.U[k])[:, :2], T(L.p[k]), T(L.phi[k]), T(L.U[k - 1])[:, :2]; pc0 = T(L.p[k - 1])
            prev_h = None
            for si, P in enumerate(out.stages):
                g, ev = P.geom, P.ev; eU = P.U - Uc
                rms = lambda x: (x.square().mean() + 1e-12).sqrt()                  # smooth at 0 (sqrt'(0) is infinite)
                h = {'fsi': (ev.R_fsi.square() + 1e-12).sqrt() / s['Fy'],
                     'cont': rms(ev.cont / ph.cell_capacity(c, g)),
                     'wall': rms(ev.wall_flux / (c.U_inf * g.A[c.rear_faces]))}
                for j in CONSTRAINTS: hs.setdefault('%s_s%d' % (j, si), []).append(float(h[j].detach()))
                if si == 0: prev_h = h; continue                                       # stage 0 = explicit guess, no parameters
                t = {'u': eU.square().sum(1).mean() / s['u'] ** 2,
                     'du': ((P.U - Uprev) - (Uc - Uc0)).square().sum(1).mean() / s['du'] ** 2,
                     'hf': (eU - ph.neighbour_mean(c, eU)).square().sum(1).mean() / s['du'] ** 2,
                     'p': (P.p - pc).square().mean() / s['p'] ** 2,
                     'phi': ((P.phi - phc) / g.A).square().mean() / s['phi'] ** 2,
                     'w': (ph.vorticity(c, g, P.U) - ph.vorticity(c, g, Uc)).square().mean() / s['w'] ** 2,
                     'y': ((P.y - float(L.y[k])) / s['y']).square(), 'v': ((P.v - float(L.v[k])) / s['v']).square(),
                     'a': ((P.a - float(L.a[k])) / s['a']).square(),
                     'kin': (ev.r_y / s['y']).square() + (ev.r_v / s['v']).square(),
                     # wall loads of the predicted flow (the domain-mean p loss barely sees the near-wall cells)
                     'Fy': ((ev.Fy - float(L.F[k, 0])) / s['Fy']).square(), 'Fp': ((ev.Fp - float(L.F[k, 1])) / s['Fp']).square(),
                     'Fv': ((ev.Fv - float(L.F[k, 2])) / s['Fv']).square(),
                     'pw': (P.p[wall_cells] - pc[wall_cells]).square().mean() / s['p'] ** 2,
                     # one-step increments: a value loss barely penalises a frozen pressure (lift changes a few % per step)
                     'dp': ((P.p - pprev) - (pc - pc0)).square().mean() / s['dp'] ** 2,
                     'dpw': ((P.p - pprev)[wall_cells] - (pc - pc0)[wall_cells]).square().mean() / s['dp'] ** 2,
                     'dFy': (((ev.Fy - Fprev) - float(L.F[k, 0] - L.F[k - 1, 0])) / s['dFy']).square(),
                     'dec': sum(torch.relu(h[j] - prev_h[j]) for j in ('fsi', 'cont'))}
                # constraints in units of their tolerance (h/tol <= 1), so that the multipliers act on residuals of any scale
                al = al_scale * sum(lam[j] * h[j] / AL['tol'][j] + .5 * AL['rho'][j] * (h[j] / AL['tol'][j]).square() for j in CONSTRAINTS)
                stage = sum(W.get(nm, 0.0) * v for nm, v in t.items()) + al
                total = total + SW[si - 1] * stage / n
                for nm, v in t.items(): acc['L_' + nm] = acc.get('L_' + nm, 0.0) + float(v.detach()) * SW[si - 1] / n
                prev_h = h
            Uprev = out.stages[-1].U; pprev = out.stages[-1].p; Fprev = out.stages[-1].ev.Fy
            fin = out.stages[-1]
            eF.append((fin.ev.Fy - float(L.F[k, 0])) / s['Fy']); ey.append((fin.y - float(L.y[k])) / s['y'])
            Fp_.append(fin.ev.Fy); Fc_.append(float(L.F[k, 0])); Pp_.append(fin.ev.Fy * fin.v); Pc_.append(float(L.F[k, 0] * L.v[k]))
            Qp_.append(fin.ev.Fy * (fin.y - y0)); Qc_.append(float(L.F[k, 0] * (L.y[k] - y0)))
            yp_.append(fin.y - y0); vp_.append(fin.v); yc_.append(float(L.y[k] - y0)); vc_.append(float(L.v[k]))
        # low-frequency bias: window-mean lift / displacement error of the final stage. Slow drift of the mean lift over
        # long rollouts is weighted by per-step terms no more than any single-step miss, so it is penalised explicitly.
        Fp_ = torch.stack(Fp_); Fc_ = Fp_.new_tensor(Fc_)
        mb = {'mFy': torch.stack(eF).mean().square(), 'my': torch.stack(ey).mean().square(),
              # lift oscillation within the window: fluctuation strength and the work done on the body (energy transfer)
              'sFy': ((Fp_.std() - Fc_.std()) / s['dFy']).square(),
              'mP': ((torch.stack(Pp_).mean() - sum(Pc_) / len(Pc_)) / (s['Fy'] * s['v'])).square(),
              # lift in phase with the displacement (added stiffness): sets the oscillation frequency, as F_y v sets the energy transfer
              'mQ': ((torch.stack(Qp_).mean() - sum(Qc_) / len(Qc_)) / (s['Fy'] * s['y'])).square(),
              # oscillation rate: phase advance of the (y, -v/w) phasor over the window relative to CFD's advance over the same steps.
              # Independent of amplitude and of the phase offset accumulated during the burn-in, so it constrains the frequency only.
              'mW': phase_rate_err(torch.stack(yp_), torch.stack(vp_), yc_, vc_, s['v'] / s['y'])}
        for nm, v in mb.items():
            total = total + W.get(nm, 0.0) * v; acc['L_' + nm] = float(v.detach())
        return total, acc, hs

    def save(u):
        torch.save(model.bundle(optimizer=optimizer.state_dict(), updates=u, lam=lam, config_train=cfg, code_tag=cfg['code_tag'],
                                scales={c.name: L.scales for c, L in zip(cases, labels)}), out / ('checkpoint_u%05d.pt' % u))
        keep, every = int(cfg.get('keep_checkpoints', 3)), int(cfg.get('milestone_every', 0))
        number = lambda p: int(p.stem.split('_u')[1])
        for p in [p for p in sorted(out.glob('checkpoint_u*.pt')) if not (every and number(p) % every == 0)][:-keep]: p.unlink()

    try:
        init = cfg.get('init'); resume = False; saved = None
        if init:
            if sha(init['path']) != init['sha256']: raise ValueError('Init checkpoint pin')
            saved = torch.load(init['path'], map_location=dev, weights_only=False)
            if saved['config']['network'] != mc['network']: raise ValueError('Init checkpoint has a different network')
            norm = Normalizer(**saved['normalizer'])
        else:
            norm = compute_normalizer(cases)
        model = FSIGraphModel(mc, norm, activation_checkpoint=bool(cfg.get('activation_checkpoint', True))).to(dev)
        if saved is not None:
            model.load_state_dict(saved['model'], strict=True)
            resume = init.get('mode') == 'resume'                   # 'warm': weights only, fresh optimizer and schedule
            if resume: lam.update(saved.get('lam', {}))
        model.set_network_dtype(); model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        if rank == 0: report['normalizer'] = norm.as_dict(); report['parameters'] = sum(p.numel() for p in params)
        if not resume:
            flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu(); dist.broadcast(flat, 0)
            torch.nn.utils.vector_to_parameters(flat.to(dev), model.parameters())
        oc = cfg['optimizer']
        optimizer = torch.optim.AdamW(params, lr=oc['lr'], betas=tuple(oc.get('betas', (0.9, 0.999))), weight_decay=oc.get('weight_decay', 0.0))
        start = 1
        if resume: optimizer.load_state_dict(saved['optimizer']); start = int(saved['updates']) + 1
        del saved
        base_lr = oc['lr']

        K = int(cfg['windows_per_rank']); G = world * K
        windows = [(ci, o) for ci, c in enumerate(cases) for o in range(0, len(c.times) - NK - Hmax, int(cfg['window_stride']))]
        per_epoch = len(windows) // G
        cond = lambda ci: (cases[ci].U_inf, cases[ci].omega)
        def epoch_order(e):
            rng = random.Random(int(cfg['seed']) * 1000 + e); gr = {}
            for w in windows: gr.setdefault(cond(w[0]), []).append(w)
            for g in gr.values(): rng.shuffle(g)
            keys = sorted(gr); rng.shuffle(keys); n = max(len(g) for g in gr.values())
            return [gr[k][i] for i in range(n) for k in keys if i < len(gr[k])]
        total_u = int(cfg['total_updates'])
        if rank == 0: report.update(windows=len(windows), updates_per_epoch=per_epoch, global_batch=G)
        hsum = {}; done = start - 1
        for u in range(start, total_u + 1):
            flag = torch.tensor([1.0 if rank == 0 and Path('/control/STOP').exists() else 0.0]); dist.broadcast(flag, 0)
            if flag.item() > 0: break
            H = horizon_at(cfg['horizon_schedule'], u); e, j = divmod(u - 1, per_epoch); order = epoch_order(e)
            batch = order[j * G:(j + 1) * G][rank * K:(rank + 1) * K]
            f = lr_factor(u, oc['warmup_updates'], total_u, oc['min_lr_ratio'])
            for g in optimizer.param_groups: g['lr'] = base_lr * f
            # data first, then constraints: multipliers off before al_start, linear ramp over al_ramp updates
            al_start, al_ramp = int(AL.get('start', 1)), max(1, int(AL.get('ramp', 1)))
            al_scale = 0.0 if u < al_start else min(1.0, (u - al_start + 1) / al_ramp)
            t0 = time.monotonic(); optimizer.zero_grad(set_to_none=True); rows = []
            faulthandler.dump_traceback_later(int(cfg.get('stall_dump_s', 900)), repeat=True)
            for wi, (ci, off) in enumerate(batch):
                tw = time.monotonic(); rs = random.Random(int(cfg['seed']) * 7919 + u * 101 + rank * 13 + wi)
                burn = rs.choice(burn_choices_at(u))
                outs = list(model.rollout(cases[ci], off, H, burn_in=burn))
                start_U = outs[0][0].U1                                                   # the state the window starts from
                loss, terms, hs = window_loss(ci, off, outs, burn, start_U)
                (loss / K).backward()
                rows.append({'case': cases[ci].name, 'offset': off, 'burn_in': burn, 'loss': float(loss.detach()), **terms,
                             **{k: float(np.mean(v)) for k, v in hs.items()}})
                print(json.dumps({'rank': rank, 'update': u, 'window': wi, 'case': cases[ci].name, 'burn_in': burn,
                                  'loss': round(rows[-1]['loss'], 5), 'wall_s': round(time.monotonic() - tw, 1),
                                  'peak_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}), file=sys.stderr, flush=True)
                del loss, outs
            faulthandler.cancel_dump_traceback_later()
            vec = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params]).double().cpu()
            dist.all_reduce(vec); vec = vec / world
            if not bool(torch.isfinite(vec).all()): raise ValueError('Nonfinite gradients')
            i0 = 0
            for p in params:
                n = p.numel(); p.grad = vec[i0:i0 + n].view_as(p).to(p); i0 += n
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip']))
            waited = 0
            while mem_gib() < cfg['minimum_available_gib']:
                time.sleep(30); waited += 30
                if waited > 1800: raise RuntimeError('Host memory below the floor for 30 min')
            optimizer.step(); torch.cuda.synchronize()
            # ---- augmented-Lagrangian multipliers (final-stage constraint means over all ranks' windows)
            fin = torch.tensor([float(np.mean([r['%s_s%d' % (jn, mc['network']['stages'])] for r in rows])) for jn in CONSTRAINTS], dtype=torch.float64)
            dist.all_reduce(fin); fin = fin / world
            for jn, v in zip(CONSTRAINTS, fin.tolist()): hsum[jn] = hsum.get(jn, []) + [v]
            if u % int(AL['every']) == 0 and u >= al_start:
                for jn in CONSTRAINTS:
                    lam[jn] = max(0.0, lam[jn] + AL['rho'][jn] * (float(np.mean(hsum[jn])) / AL['tol'][jn] - 1.0))
                hsum = {}
            gathered = [None] * world
            dist.all_gather_object(gathered, {'rows': rows, 'peak_GB': torch.cuda.max_memory_allocated() / 1e9})
            torch.cuda.reset_peak_memory_stats()
            if rank == 0:
                allr = [r for g in gathered for r in g['rows']]
                row = {'update': u, 'epoch': e, 'H': H, 'lr_factor': f, 'loss': float(np.mean([r['loss'] for r in allr])),
                       'terms': {k: float(np.mean([r[k] for r in allr])) for k in allr[0] if k.startswith('L_')},
                       'constraints_by_stage': {k: float(np.mean([r[k] for r in allr])) for k in allr[0] if k.split('_s')[0] in CONSTRAINTS},
                       'lambda': dict(lam), 'al_scale': al_scale, 'grad_norm': gnorm, 'windows': len(allr), 'wall_s': time.monotonic() - t0,
                       'peak_GB_max': max(g['peak_GB'] for g in gathered), 'mem_available_gib': mem_gib(),
                       'utc': datetime.now(timezone.utc).isoformat()}
                with (out / 'progress.jsonl').open('a') as fh: fh.write(json.dumps(row) + '\n')
                print(json.dumps({k: row[k] for k in ('update', 'H', 'loss', 'grad_norm', 'wall_s', 'windows')}), flush=True)
                if u % cfg['checkpoint_every'] == 0 or u == total_u: save(u)
            done = u
        if rank == 0:
            if done >= start and not (out / ('checkpoint_u%05d.pt' % done)).exists(): save(done)
            report['status'] = 'stopped' if done < total_u else 'completed'; report['last_completed_update'] = done
    except Exception as err:
        if rank == 0: report['status'] = 'failed'; report['error'] = repr(err); report['traceback'] = traceback.format_exc()
        raise
    finally:
        if rank == 0:
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            report['source_unchanged'] = all(sha(Path('/work') / k) == v for k, v in manifest.items())
            (out / 'report.json').write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
