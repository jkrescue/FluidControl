"""Error diagnosis of an FSIGraphNet checkpoint on evaluation cases (container: torch + the released source on PYTHONPATH).

  python diagnose.py <checkpoint.pt> <views_dir> <split.json> <out_dir> <role>[,<role>...] [steps=300] [offset=0]

Per case, two passes of the same model (no gradient):
  autonomous   the normal rollout from the known frames (as fsi_mgn_eval.py)
  one-step     at every step of the same range the model starts from the CFD state (teacher forcing) and predicts one step
Recorded per step (CFD labels are read only for the comparison):
  structure    y, v, a; the lift Fy* used by the final stage's equation of motion; a_eq (Newmark with Fy*) and the network's
               acceleration correction a - a_eq, as an equivalent force F_corr = (m + c dt/2 + k dt^2/4)(a - a_eq)
  loads        final Fy, Fp, Fv of the prediction and of CFD; rear-wall cell pressure of both
  field        per region (near rear cylinder, gap, near wake, far wake, rest): squared errors of u, v, p, CFD fluctuation
               energy, enstrophy of both, grid-scale (cell minus neighbour mean) energy of both, grid-scale and smooth parts
               of the velocity error
Output: <out_dir>/<case>.npz. HELDOUT is refused.
"""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

from fsi_mgn.model import FSIGraphModel
from fsi_mgn.data import Case
from fsi_mgn import physics as ph

REGIONS = ('near_rear', 'gap', 'near_wake', 'far_wake', 'rest')


def regions(c, xy0):
    C = xy0[np.array(c.z('cell_vertices_ccw'))].mean(1); lab = np.array(c.z('node_types')).reshape(-1)
    rc = xy0[lab == 5].mean(0); fc = xy0[lab == 4].mean(0); D = c.D
    dr = np.linalg.norm(C - rc, axis=1) / D; x, y = (C[:, 0] - rc[0]) / D, (C[:, 1] - rc[1]) / D
    m = {'near_rear': dr < 1.5,
         'gap': (C[:, 0] > fc[0]) & (C[:, 0] < rc[0]) & (np.abs(C[:, 1] - fc[1]) / D < 1.5) & (dr >= 1.5),
         'near_wake': (x >= 0) & (x < 8) & (np.abs(y) < 3) & (dr >= 1.5),
         'far_wake': (x >= 8) & (np.abs(y) < 4)}
    taken = np.zeros(len(C), bool)
    for k in m: m[k] = m[k] & ~taken; taken |= m[k]
    m['rest'] = ~taken
    return {k: torch.as_tensor(v, device=c.dev) for k, v in m.items()}


def main(ck, views, split, out, roles, steps=300, offset=0):
    roles = roles.split(',')
    if 'HELDOUT' in roles: raise SystemExit('HELDOUT is for the final acceptance only')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model, _ = FSIGraphModel.load(ck, 'cuda'); model.eval(); NK = model.known_frames; dt = model.dt
    out = Path(out); out.mkdir(parents=True, exist_ok=True); sp = json.loads(Path(split).read_text())
    for role in roles:
        for cd in sp['roles'][role]:
            c = Case(Path(views) / cd['name'], cd['sequence_sha256'], cd['report_sha256'], 'cuda')
            n = min(steps, len(c.times) - (offset + NK)); a0 = offset + NK - 1
            Ul, Pl, XY = c.z('native_cell_U'), c.z('native_cell_p'), c.z('node_xy'); F = np.array(c.z('native_force_components'))
            Uall = np.array(Ul)[..., :2]; Um = torch.as_tensor(Uall.mean(0), device='cuda'); Pm = torch.as_tensor(np.array(Pl).mean(0), device='cuda')
            R = regions(c, np.array(XY[a0])); meff = c.mass + .5 * c.damping * dt + .25 * c.stiffness * dt * dt
            rc = c.rear_cells; w = c.morph_weights(offset, NK)
            T = lambda x: torch.as_tensor(np.array(x), dtype=torch.float64, device='cuda')

            def struct_terms(st, out_):
                Pf, Pq = out_.stages[-1], out_.stages[-2]
                Fp_, Fv_, _ = ph.wall_force(c, Pq.geom, Pf.U, Pf.p, Pq.v); Fy_ = Fp_ + Fv_
                aeq = (Fy_ - c.damping * (st.v + .5 * dt * st.a) - c.stiffness * (st.y + dt * st.v + .25 * dt * dt * st.a)) / meff
                return [float(Pf.y), float(Pf.v), float(Pf.a), float(Fy_), float(aeq), float(meff * (Pf.a - aeq)),
                        float(Pf.ev.Fy), float(Pf.ev.Fp), float(Pf.ev.Fv)]

            def field_terms(P, k):
                Uc, pc = T(Ul[k])[:, :2], T(Pl[k]); g = P.geom; e = P.U - Uc
                wm, wc = ph.vorticity(c, g, P.U), ph.vorticity(c, g, Uc)
                hm = (P.U - ph.neighbour_mean(c, P.U)).square().sum(1); hc = (Uc - ph.neighbour_mean(c, Uc)).square().sum(1)
                enm = ph.neighbour_mean(c, e); ehp = (e - enm).square().sum(1); elp = enm.square().sum(1)
                q = {'eu': e[:, 0].square(), 'ev': e[:, 1].square(), 'ep': (P.p - pc).square(), 'cu': (Uc[:, 0] - Um[:, 0]).square(),
                     'cv': (Uc[:, 1] - Um[:, 1]).square(), 'cp': (pc - Pm).square(), 'ensm': wm.square(), 'ensc': wc.square(),
                     'hfm': hm, 'hfc': hc, 'ehp': ehp, 'elp': elp}
                return np.array([[float(q[kk][R[r]].mean()) for kk in q] for r in REGIONS]), \
                    P.p[rc].float().cpu().numpy(), pc[rc].float().cpu().numpy()

            res = {}
            with torch.no_grad():
                # ---- autonomous rollout (identical to model.rollout, keeping the previous state for the structure terms)
                st = c.initial_state(offset, NK); st = replace(st, mem=model._warm_memory(c, offset, w))
                S, Fd, PW, PC = [], [], [], []
                for i in range(n):
                    new, out_ = model.step(c, st, w); k = a0 + 1 + i
                    S.append(struct_terms(st, out_)); f, pm, pcw = field_terms(out_.stages[-1], k); Fd.append(f); PW.append(pm); PC.append(pcw); st = new
                res.update(auto_struct=np.array(S), auto_field=np.array(Fd), auto_pwall=np.array(PW), cfd_pwall=np.array(PC))
                # ---- one-step from the CFD state at every step of the same range (memory warmed from known frames as usual)
                S, Fd = [], []
                for i in range(n):
                    off = offset + i; wi = c.morph_weights(off, NK)
                    st = c.initial_state(off, NK); st = replace(st, mem=model._warm_memory(c, off, wi))
                    _, out_ = model.step(c, st, wi); k = off + NK
                    S.append(struct_terms(st, out_)); Fd.append(field_terms(out_.stages[-1], k)[0])
                res.update(one_struct=np.array(S), one_field=np.array(Fd))
            fr = c.frames; idx = np.arange(a0 + 1, a0 + 1 + n)
            res.update(cfd_y=np.array([fr[k]['centreOfRotation'][1] for k in idx]), cfd_v=np.array([fr[k]['velocity'][1] for k in idx]),
                       cfd_a=np.array([fr[k]['acceleration'][1] for k in idx]), cfd_F=F[idx], cfd_y_all=np.array([f['centreOfRotation'][1] for f in fr]),
                       mass=c.mass, damping=c.damping, stiffness=c.stiffness, dt=dt, U_inf=c.U_inf, omega=c.omega,
                       struct_cols=np.array(['y', 'v', 'a', 'Fy_used', 'a_eq', 'F_corr', 'Fy', 'Fp', 'Fv']),
                       field_cols=np.array(['eu', 'ev', 'ep', 'cu', 'cv', 'cp', 'ensm', 'ensc', 'hfm', 'hfc', 'ehp', 'elp']),
                       regions=np.array(REGIONS), region_cells=np.array([int(R[r].sum()) for r in REGIONS]))
            np.savez(out / (c.name + '.npz'), **res); print('done', c.name, n, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:6], *[int(x) for x in sys.argv[6:8]])
