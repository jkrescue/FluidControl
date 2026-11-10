#!/usr/bin/env python3
"""Render an evaluation run (fsi_mgn_eval.py) against CFD.

  python3 tools/render.py <eval_run_dir> <case_view_dir> <out_dir>

Figures: field_{u,v,p,w}.png (CFD | prediction | prediction - CFD on the native cells at selected steps), curves.png
(rear-cylinder y, Fy, Fp, Fv), errors.png (per-step relative RMS and correlation).
Host numpy + matplotlib only. Labels are read here, after the evaluation finished.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

run, view, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
case = view.name; m = json.loads((run / 'results/metrics.json').read_text())
r0 = [r for r in m['runs'] if r['case'] == case][0]; off = r0['offset']; NK = int(m['model_config']['known_frames']); a = off + NK - 1
MODEL = 'FSIGraphNet'
d = run / 'results' / case / ('o%03d' % off)
traj = json.loads((d / 'trajectory.json').read_text())['rows']; fr = np.load(d / 'frames.npz'); n = len(traj)
z = np.load(view / 'sequence.npz'); quads = z['cell_vertices_ccw']; nc = len(quads)
tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]); cell_of_tri = np.concatenate([np.arange(nc), np.arange(nc)])
own = z['planar_face_owner']; nei = z['planar_face_neighbor']; inner = nei >= 0; nei_ = np.where(inner, nei, own)


def vort(U, S, V):
    Uf = np.where(inner[:, None], .5 * (U[own] + U[nei_]), U[own]); cz = S[:, 0] * Uf[:, 1] - S[:, 1] * Uf[:, 0]
    w = np.zeros(nc); np.add.at(w, own, cz); np.add.at(w, nei[inner], -cz[inner]); return w / V


def cfd(i):
    k = a + i; U = z['native_cell_U'][k][:, :2]
    return {'u': U[:, 0], 'v': U[:, 1], 'p': z['native_cell_p'][k], 'w': vort(U, z['face_sf'][k][:, :2], z['cell_volume'][k])}, z['node_xy'][k]


def pred(i):
    U = fr['U'][i - 1].astype(float)
    return {'u': U[:, 0], 'v': U[:, 1], 'p': fr['p'][i - 1].astype(float), 'w': fr['w'][i - 1].astype(float)}, fr['xy'][i - 1].astype(float)


P = r0['period_steps']; steps = sorted({1, 10, int(round(P / 2)), int(round(P)), int(round(1.5 * P)), n} & set(range(1, n + 1)))
label = {'u': 'u (m/s)', 'v': 'v (m/s)', 'p': 'p (kinematic, m²/s²)', 'w': 'vorticity ω_z (1/s)'}
for key in ('u', 'v', 'p', 'w'):
    fig, ax = plt.subplots(len(steps), 3, figsize=(16, 2.3 * len(steps)), constrained_layout=True)
    for r, i in enumerate(steps):
        fc, xc = cfd(i); fp, xp = pred(i); ref = fc[key]
        lo, hi = np.percentile(ref, [1, 99])
        vmin, vmax = (lo, hi) if key == 'u' else ((-1.5, 1.5) if key == 'w' else (-max(abs(lo), abs(hi)), max(abs(lo), abs(hi))))
        for c, (xy, val, ttl) in enumerate(((xc, ref, 'CFD'), (xp, fp[key], MODEL))):
            im = ax[r, c].tripcolor(mtri.Triangulation(xy[:, 0], xy[:, 1], tris), facecolors=val[cell_of_tri], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax[r, c].set_title('%s  step %d (t+%.1f s)' % (ttl, i, 0.1 * i), fontsize=9)
        fig.colorbar(im, ax=ax[r, :2], shrink=0.9, label=label[key])
        e = fp[key] - ref; em = 0.5 * np.percentile(abs(ref - ref.mean()), 99) if key != 'w' else 0.75
        im2 = ax[r, 2].tripcolor(mtri.Triangulation(xc[:, 0], xc[:, 1], tris), facecolors=e[cell_of_tri], cmap='PuOr_r', vmin=-em, vmax=em)
        ax[r, 2].set_title(MODEL + ' − CFD   rel.RMS %.2f' % np.sqrt((e ** 2).mean() / ((ref - ref.mean()) ** 2).mean()), fontsize=9)
        fig.colorbar(im2, ax=ax[r, 2], shrink=0.9)
        for c in range(3): ax[r, c].set_xlim(-2, 16); ax[r, c].set_ylim(-3, 3); ax[r, c].set_aspect('equal'); ax[r, c].tick_params(labelsize=7)
    fig.suptitle('%s — %s, autonomous strong-FSI rollout from 20 known frames (no CFD reset)' % (label[key], case), fontsize=11)
    fig.savefig(out / ('field_%s.png' % key), dpi=100); plt.close(fig)

t = 0.1 * np.arange(1, n + 1); idx = a + np.arange(1, n + 1); F = z['native_force_components']
fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
for k, (name, key, cv) in enumerate((('rear cylinder y (m)', 'y', z['native_y_absolute'][idx]), ('Fy total (N)', 'Fy', F[idx, 0]),
                                      ('Fy pressure (N)', 'Fp', F[idx, 1]), ('Fy viscous (N)', 'Fv', F[idx, 2]))):
    ax[k].plot(t, cv, 'k-', lw=1.8, label='CFD'); ax[k].plot(t, [q[key] for q in traj], 'C3-', lw=1.4, label=MODEL)
    ax[k].set_ylabel(name); ax[k].grid(alpha=.3)
    for c in np.arange(P, n + 1, P) * 0.1: ax[k].axvline(c, color='0.75', lw=0.8, ls=':')
ax[0].legend(fontsize=9); ax[-1].set_xlabel('time after the last known frame (s); dotted = CFD periods')
fig.suptitle('Structure motion and loads — ' + case); fig.savefig(out / 'curves.png', dpi=100); plt.close(fig)

ps = r0['per_step']; fig, axs = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
for k in ('u', 'v', 'p'): axs[0].plot(t, ps[k], label=k); axs[1].plot(t, ps['corr_' + k], label=k)
axs[0].set_title('relative RMS error of native cell fields (vs CFD fluctuation)'); axs[1].set_title('field correlation with CFD'); axs[1].set_ylim(-1, 1)
for x in axs: x.grid(alpha=.3); x.legend(); x.set_xlabel('time after the last known frame (s)')
fig.savefig(out / 'errors.png', dpi=100); plt.close(fig)
json.dump({'case': case, 'steps': n, 'period_steps': P, 'figure_steps': steps, 'cycles': r0['cycles'], 'step1_du': ps['step1_du'],
           'checkpoint_updates': m.get('model_updates')}, open(out / 'summary.json', 'w'), indent=1)
print('rendered', case, n, 'steps')
