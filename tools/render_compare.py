#!/usr/bin/env python3
"""Compare several evaluation runs (fsi_mgn_eval.py, frames saved) against CFD on one case.

  python3 tools/render_compare.py <case_view_dir> <out_dir> <label>=<eval_run_dir> [<label>=<eval_run_dir> ...]

Figures:
  compare_field_{p,u,v,w}.png  rows = rollout steps, columns = CFD | model 1 | model 2 | ...; one colour scale per row (from CFD)
  compare_fields_summary.png   rows = p, u, v, vorticity; columns = CFD | models at half a period and two periods
  compare_loads.png            rear-cylinder y and Fy (total, pressure, viscous) against CFD
  compare_errors.png           per-step relative RMS error and correlation of u, v, p
All runs must use the same case, offset and step count. Host numpy + matplotlib only.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

view, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); case = view.name
runs = []
for arg in sys.argv[3:]:
    lab, d = arg.split('=', 1); d = Path(d); m = json.loads((d / 'results/metrics.json').read_text())
    r0 = [r for r in m['runs'] if r['case'] == case][0]; od = d / 'results' / case / ('o%03d' % r0['offset'])
    runs.append({'label': lab, 'm': m, 'r0': r0, 'traj': json.loads((od / 'trajectory.json').read_text())['rows'],
                 'fr': np.load(od / 'frames.npz')})
r0 = runs[0]['r0']; NK = int(runs[0]['m']['model_config']['known_frames']); a = r0['offset'] + NK - 1; n = len(runs[0]['traj'])
P = r0['period_steps']
z = np.load(view / 'sequence.npz'); quads = z['cell_vertices_ccw']; nc = len(quads)
tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]); cell_of_tri = np.concatenate([np.arange(nc), np.arange(nc)])
own = z['planar_face_owner']; nei = z['planar_face_neighbor']; inner = nei >= 0; nei_ = np.where(inner, nei, own)


def vort(U, S, V):
    Uf = np.where(inner[:, None], .5 * (U[own] + U[nei_]), U[own]); cz = S[:, 0] * Uf[:, 1] - S[:, 1] * Uf[:, 0]
    w = np.zeros(nc); np.add.at(w, own, cz); np.add.at(w, nei[inner], -cz[inner]); return w / V


def cfd(i):
    k = a + i; U = z['native_cell_U'][k][:, :2]
    return {'u': U[:, 0], 'v': U[:, 1], 'p': z['native_cell_p'][k], 'w': vort(U, z['face_sf'][k][:, :2], z['cell_volume'][k])}, z['node_xy'][k]


def pred(fr, i):
    U = fr['U'][i - 1].astype(float)
    return {'u': U[:, 0], 'v': U[:, 1], 'p': fr['p'][i - 1].astype(float), 'w': fr['w'][i - 1].astype(float)}, fr['xy'][i - 1].astype(float)


steps = sorted({int(round(P / 2)), int(round(P)), int(round(2 * P)), n} & set(range(1, n + 1)))
label = {'u': 'u (m/s)', 'v': 'v (m/s)', 'p': 'p (kinematic, m²/s²)', 'w': 'vorticity ω_z (1/s)'}
cols = len(runs) + 1
for key in ('p', 'u', 'v', 'w'):
    fig, ax = plt.subplots(len(steps), cols, figsize=(4.6 * cols, 1.75 * len(steps) + 0.6), constrained_layout=True, squeeze=False)
    for r, i in enumerate(steps):
        fc, xc = cfd(i); ref = fc[key]; lo, hi = np.percentile(ref, [1, 99])
        vmin, vmax = (lo, hi) if key == 'u' else ((-1.5, 1.5) if key == 'w' else (-max(abs(lo), abs(hi)), max(abs(lo), abs(hi))))
        panels = [(xc, ref, 'CFD')]
        for run in runs:
            fp, xp = pred(run['fr'], i); e = fp[key] - ref
            panels.append((xp, fp[key], '%s   rel.RMS %.2f' % (run['label'], np.sqrt((e ** 2).mean() / ((ref - ref.mean()) ** 2).mean()))))
        for c, (xy, val, ttl) in enumerate(panels):
            im = ax[r, c].tripcolor(mtri.Triangulation(xy[:, 0], xy[:, 1], tris), facecolors=val[cell_of_tri], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax[r, c].set_title(ttl, fontsize=9); ax[r, c].set_xlim(-2, 16); ax[r, c].set_ylim(-3, 3); ax[r, c].set_aspect('equal')
            ax[r, c].tick_params(labelsize=6)
        ax[r, 0].set_ylabel('step %d\nt+%.1f s' % (i, 0.1 * i), fontsize=9)
        fig.colorbar(im, ax=ax[r, :], shrink=0.9, pad=0.01).ax.tick_params(labelsize=7)
    fig.suptitle('%s — %s, %d-step autonomous rollout from %d known frames (no CFD reset)' % (label[key], case, n, NK), fontsize=11)
    fig.savefig(out / ('compare_field_%s.png' % key), dpi=90); plt.close(fig)

sel = [int(round(P / 2)), int(round(2 * P))]; keys = ('p', 'u', 'v', 'w'); cols = len(sel) * (len(runs) + 1)
fig, ax = plt.subplots(len(keys), cols, figsize=(3.3 * cols, 1.5 * len(keys) + 0.5), constrained_layout=True, squeeze=False)
ref0 = cfd(sel[0])[0]; lim = {}
for key in keys:
    lo, hi = np.percentile(ref0[key], [1, 99])
    lim[key] = (lo, hi) if key == 'u' else ((-1.5, 1.5) if key == 'w' else (-max(abs(lo), abs(hi)), max(abs(lo), abs(hi))))
for s, i in enumerate(sel):
    fc, xc = cfd(i); fps = [pred(run['fr'], i) for run in runs]
    for r, key in enumerate(keys):
        ref = fc[key]; vmin, vmax = lim[key]
        for j, (xy, val, ttl) in enumerate([(xc, ref, 'CFD')] + [(xp, fp[key], run['label']) for (fp, xp), run in zip(fps, runs)]):
            x = ax[r, s * (len(runs) + 1) + j]
            im = x.tripcolor(mtri.Triangulation(xy[:, 0], xy[:, 1], tris), facecolors=val[cell_of_tri], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            x.set_xlim(-2, 16); x.set_ylim(-3, 3); x.set_aspect('equal'); x.set_xticks([]); x.set_yticks([])
            if r == 0: x.set_title('%s, t+%.1f s' % (ttl, 0.1 * i), fontsize=9)
            if j == 0 and s == 0: x.set_ylabel(label[key], fontsize=8)
        if s == len(sel) - 1: fig.colorbar(im, ax=ax[r, :], shrink=0.9, pad=0.01).ax.tick_params(labelsize=7)
fig.savefig(out / 'compare_fields_summary.png', dpi=110); plt.close(fig)

t = 0.1 * np.arange(1, n + 1); idx = a + np.arange(1, n + 1); F = z['native_force_components']
fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
for k, (name, key, cv) in enumerate((('rear cylinder y (m)', 'y', z['native_y_absolute'][idx]), ('lift Fy total (N)', 'Fy', F[idx, 0]),
                                      ('Fy pressure part (N)', 'Fp', F[idx, 1]), ('Fy viscous part (N)', 'Fv', F[idx, 2]))):
    ax[k].plot(t, cv, 'k-', lw=2.2, label='CFD')
    for j, run in enumerate(runs): ax[k].plot(t, [q[key] for q in run['traj']], color='C%d' % j, lw=1.3, label=run['label'])
    ax[k].set_ylabel(name); ax[k].grid(alpha=.3)
    for c in np.arange(P, n + 1, P) * 0.1: ax[k].axvline(c, color='0.75', lw=0.8, ls=':')
ax[0].legend(fontsize=9, ncol=len(runs) + 1, loc='upper left'); ax[-1].set_xlabel('time after the last known frame (s); dotted = CFD periods')
fig.suptitle('Rear-cylinder motion and lift — ' + case); fig.savefig(out / 'compare_loads.png', dpi=90); plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True, constrained_layout=True)
for c, k in enumerate(('u', 'v', 'p')):
    for j, run in enumerate(runs):
        ps = run['r0']['per_step']; axs[0, c].plot(t, ps[k], color='C%d' % j, label=run['label']); axs[1, c].plot(t, ps['corr_' + k], color='C%d' % j)
    axs[0, c].set_title('relative RMS error, %s' % k); axs[1, c].set_title('correlation with CFD, %s' % k); axs[1, c].set_ylim(-1, 1)
    axs[1, c].set_xlabel('time after the last known frame (s)')
for x in axs.flat: x.grid(alpha=.3)
axs[0, 0].legend(fontsize=9); fig.savefig(out / 'compare_errors.png', dpi=90); plt.close(fig)
print('rendered', case, [r['label'] for r in runs], 'steps', steps)
