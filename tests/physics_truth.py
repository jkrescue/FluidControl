"""Physics formulas of fsi_mgn evaluated on CFD truth (frames of one case): the residual floors must be small.

  PYTHONPATH=src python -B tests/physics_truth.py '<case json>'      (cases under /views)

Checks: Fp, Fv, Fy vs native_force_components; R_fsi = m a + c v + k y - Fy; Newmark r_y, r_v; continuity of the CFD face flux;
rear-wall flux - mesh flux; interp(U).S - CFD flux (the gap the network's face head must learn)."""
import json, sys
import numpy as np, torch
from fsi_mgn.data import Case
from fsi_mgn import physics as ph
cd = json.loads(sys.argv[1]); c = Case('/views/' + cd['name'], cd['sequence_sha256'], cd['report_sha256'], 'cuda')
w = c.morph_weights(40, 20)   # known translation map (identified once; the same map holds for every frame)
z = c.z; T = lambda x: torch.as_tensor(np.array(x), dtype=torch.float64, device='cuda'); xy = z('node_xy')
F = np.array(z('native_force_components')); rows = []
for k in range(3, len(c.times) - 1, 7):
    g = ph.geometry(c, T(xy[k - 1]), T(xy[k]), 0.1, w)
    y, v, a = c.body(k); yn, vn, an = c.body(k - 1)
    U = T(z('native_cell_U')[k])[:, :2]; p = T(z('native_cell_p')[k]); phi = T(z('absolute_flux')[k])
    ev = ph.evaluate(c, g, U, p, phi, y, v, a, yn, vn, an, 0.1)
    flux_scale = float((c.U_inf * g.A).mean())
    rows.append([float(ev.Fp) - F[k, 1], float(ev.Fv) - F[k, 2], float(ev.Fy) - F[k, 0], float(ev.R_fsi), float(ev.r_y), float(ev.r_v),
                 float(ev.cont.abs().max()) / flux_scale, float(ev.wall_flux.abs().max()) / flux_scale,
                 float(ev.flux_gap.abs().mean()) / flux_scale, float(ev.Fy)])
r = np.array(rows); sF = F[:, 0].std()
print('case', c.name, 'frames checked', len(r), 'Fy std %.4e N' % sF)
for i, name in enumerate(['Fp - CFD', 'Fv - CFD', 'Fy - CFD', 'R_fsi', 'Newmark r_y [m]', 'Newmark r_v [m/s]']):
    print('%-18s max |.| = %.3e   (/ Fy std = %.2e)' % (name, np.abs(r[:, i]).max(), np.abs(r[:, i]).max() / sF if i in (0, 1, 2, 3) else float('nan')))
print('continuity max |sum phi| / (U_inf |S|)           = %.2e' % r[:, 6].max())
print('rear wall  max |phi - mesh(v)| / (U_inf |S|)       = %.2e' % r[:, 7].max())
print('interp(U).S - phi  mean / (U_inf |S|)          = %.2e' % r[:, 8].mean())
