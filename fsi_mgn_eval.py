"""Evaluation of an FSIGraphNet checkpoint: autonomous prediction, then metrics against CFD.

Phase 1 (prediction): for every case and start offset the known frames up to the anchor build the initial state and the
model (same model.rollout as in training, no gradient) predicts `steps` steps, one forward pass each. No CFD reset and no
labels. Recorded per step: wall time and the physics consistency of every coupling stage.
Phase 2 (metrics, labels opened only now): per step, relative RMS error (vs CFD fluctuation) and correlation of cell
u, v, p; per cycle, structural-response metrics (fsi_mgn.metrics.viv_cycles).
Container mounts: /work, /contracts/config.json, /views, /ckpt/checkpoint.pt, /results.
config: {cases: [{name, sequence_sha256, report_sha256, role}], offsets: [int] | starts: int, steps: int,
         save_frames: [case names], model_checkpoint_sha256}
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, time, traceback


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    cfg = json.loads(Path('/contracts/config.json').read_text()); out = Path('/results')
    manifest = json.loads(Path('/work/SOURCE_MANIFEST.json').read_text())
    if any(sha(Path('/work') / k) != v for k, v in manifest.items()): raise ValueError('Source pins')
    import numpy as np, torch, faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    from fsi_mgn.model import FSIGraphModel
    from fsi_mgn.data import Case
    from fsi_mgn import physics as ph
    from fsi_mgn.metrics import viv_cycles
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8': raise ValueError('Deterministic runtime required (CUBLAS_WORKSPACE_CONFIG)')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    torch.manual_seed(0); torch.set_num_threads(2)
    rep = {'config': cfg, 'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        ck = '/ckpt/checkpoint.pt'
        if sha(ck) != cfg['model_checkpoint_sha256']: raise ValueError('Checkpoint pin')
        model, saved = FSIGraphModel.load(ck, 'cuda'); model.eval(); op = model.config
        rep.update(model_config=op, model_updates=saved.get('updates')); del saved
        NK = int(op['known_frames']); runs = []
        # ---------------- phase 1: predictions
        for cd in cfg['cases']:
            c = Case(Path('/views') / cd['name'], cd['sequence_sha256'], cd['report_sha256'], 'cuda')
            avail = len(c.times) - NK
            offs = cfg.get('offsets') or [int(round(i * (avail - cfg['steps']) / max(1, cfg['starts'] - 1))) for i in range(cfg['starts'])]
            for off in offs:
                steps = min(int(cfg['steps']), len(c.times) - (off + NK))
                rows = []; U, P, X, W = [], [], [], []; status = 'completed'; t0 = time.monotonic()
                with torch.no_grad():
                    it = model.rollout(c, off, steps)   # THE model, same call as training (one forward pass per step)
                    for i in range(1, steps + 1):
                        ts = time.monotonic()
                        try:
                            st, out_ = next(it)
                        except ValueError as err:          # e.g. invalid (inverted) cell geometry from a diverged prediction
                            status = 'failed_step_%d: %s' % (i, repr(err)[:200]); break
                        torch.cuda.synchronize(); step_s = time.monotonic() - ts
                        P_ = out_.stages[-1]; ev = P_.ev; g = P_.geom
                        rows.append({'step': i, 'time_s': st.time_s, 'y': float(st.y), 'v': float(st.v), 'a': float(st.a), 'Fy': float(ev.Fy),
                                     'Fp': float(ev.Fp), 'Fv': float(ev.Fv), 'step_wall_s': step_s,
                                     'R_fsi_by_stage': [float(q.ev.R_fsi) for q in out_.stages],
                                     'cont_rms_by_stage': [float((q.ev.cont / ph.cell_capacity(c, q.geom)).square().mean().sqrt()) for q in out_.stages],
                                     'wall_rms': float((ev.wall_flux / (c.U_inf * g.A[c.rear_faces])).square().mean().sqrt())})
                        U.append(st.U.float().cpu().numpy()); P.append(st.p.float().cpu().numpy()); X.append(st.points.float().cpu().numpy())
                        W.append(ph.vorticity(c, g, st.U).float().cpu().numpy())
                d = out / cd['name'] / ('o%03d' % off); d.mkdir(parents=True, exist_ok=True)
                (d / 'trajectory.json').write_text(json.dumps({'case': cd['name'], 'offset': off, 'status': status, 'rows': rows,
                                                                'wall_s': time.monotonic() - t0}))
                if cd['name'] in cfg.get('save_frames', []) and off == offs[0]:
                    np.savez(d / 'frames.npz', U=np.array(U), p=np.array(P), w=np.array(W), xy=np.array(X))
                runs.append({'case': c, 'role': cd.get('role'), 'offset': off, 'rows': rows, 'U': U, 'P': P, 'status': status})
                print(cd['name'], off, status, len(rows), flush=True)
        # ---------------- phase 2: metrics (labels opened now)
        rep['runs'] = []
        for r in runs:
            c = r['case']; a = r['offset'] + NK - 1; n = len(r['rows'])
            Ul = c.z('native_cell_U'); Pl = c.z('native_cell_p'); F = np.array(c.z('native_force_components')); Y = np.array(c.z('native_y_absolute'))
            Uall = np.array(Ul)[..., :2]; Umean = Uall.mean(0); su = np.sqrt(((Uall - Umean) ** 2).mean(axis=(0, 1)))
            pall = np.array(Pl); pmean = pall.mean(0); sp = float(np.sqrt(((pall - pmean) ** 2).mean()))
            per = {'u': [], 'v': [], 'p': [], 'corr_u': [], 'corr_v': [], 'corr_p': [], 'step1_du': None}
            corr = lambda x, y: float(((x - x.mean()) * (y - y.mean())).sum() / np.sqrt(((x - x.mean()) ** 2).sum() * ((y - y.mean()) ** 2).sum()))
            for i in range(n):
                k = a + 1 + i; Uc = np.array(Ul[k])[:, :2]; pc = np.array(Pl[k]); Up = r['U'][i]; Pp = r['P'][i]
                per['u'].append(float(np.sqrt(((Up[:, 0] - Uc[:, 0]) ** 2).mean()) / su[0])); per['v'].append(float(np.sqrt(((Up[:, 1] - Uc[:, 1]) ** 2).mean()) / su[1]))
                per['p'].append(float(np.sqrt(((Pp - pc) ** 2).mean()) / sp))
                per['corr_u'].append(corr(Up[:, 0] - Umean[:, 0], Uc[:, 0] - Umean[:, 0])); per['corr_v'].append(corr(Up[:, 1] - Umean[:, 1], Uc[:, 1] - Umean[:, 1]))
                per['corr_p'].append(corr(Pp - pmean, pc - pmean))
            if n:
                Uc1 = np.array(Ul[a + 1])[:, :2]; Uc0 = np.array(Ul[a])[:, :2]
                per['step1_du'] = float(np.sqrt(((r['U'][0] - Uc1) ** 2).sum(-1).mean() / ((Uc1 - Uc0) ** 2).sum(-1).mean()))
            y = np.array([q['y'] for q in r['rows']]); fy = np.array([q['Fy'] for q in r['rows']])
            if n: per['step_wall_s_mean'] = float(np.mean([q['step_wall_s'] for q in r['rows'][1:]] or [r['rows'][0]['step_wall_s']]))
            yc = Y[a + 1:a + 1 + n]; fc = F[a + 1:a + 1 + n, 0]
            s = np.where(np.diff(np.sign(Y - Y.mean())) != 0)[0]; P = 2 * (s[-1] - s[0]) / (len(s) - 1)
            cyc = viv_cycles(y, fy, yc, fc, model.dt, y_mean=float(Y.mean()))
            rep['runs'].append({'case': c.name, 'role': r['role'], 'offset': r['offset'], 'status': r['status'], 'steps': n, 'period_steps': P,
                                'per_step': per, 'cycles': cyc})
            print(c.name, r['offset'], json.dumps(cyc), flush=True)
        rep['status'] = 'completed'
    except Exception as e:
        rep['status'] = 'failed'; rep['error'] = repr(e); rep['traceback'] = traceback.format_exc(); raise
    finally:
        rep['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (out / 'metrics.json').write_text(json.dumps(rep, indent=1, default=float))


if __name__ == '__main__':
    main()
