"""Per-cycle structural-response metrics of a predicted VIV trajectory against the reference (numpy only).

Cycles run from one upward zero crossing of the reference displacement (about its mean) to the next.
  amplitude_ratio  half peak-to-peak of y / reference                    frequency_ratio  T_ref / T
  phase_deg        zero-crossing time lag x 360 / T_ref (+ = lagging)    y_corr           corr(y, y_ref)
  lift_mean_err    (mean Fy - mean Fy_ref) / Fy_ref amplitude            lift_rms_ratio   std(Fy) / std(Fy_ref)
  lift_phase_err   lift-displacement phase at the fundamental minus the reference value [deg]
  energy_err       (cycle work of Fy on the body - reference) / reference integral of |Fy dy/dt|
"""
import numpy as np


def _up(x):
    return np.where((x[:-1] < 0) & (x[1:] >= 0))[0]


def viv_cycles(y, fy, y_ref, fy_ref, dt, y_mean=None):
    n = min(len(y), len(y_ref)); y, fy, y_ref, fy_ref = (np.asarray(q, float)[:n] for q in (y, fy, y_ref, fy_ref))
    m = y_ref.mean() if y_mean is None else y_mean
    zr, zp = _up(y_ref - m), _up(y - m)
    v, v_ref = np.gradient(y, dt), np.gradient(y_ref, dt)
    half = lambda x: (x.max() - x.min()) / 2
    out = []
    for i in range(len(zr) - 1):
        s = slice(zr[i], zr[i + 1]); T = dt * (zr[i + 1] - zr[i]); t = dt * np.arange(zr[i + 1] - zr[i])
        ph = lambda x: np.angle(np.sum((x - x.mean()) * np.exp(-2j * np.pi * t / T)))
        dphi = lambda f, q: np.angle(np.exp(1j * (ph(f) - ph(q))))
        nxt = zp[zp >= zr[i] - (zr[i + 1] - zr[i]) // 2]
        T_p = dt * (nxt[1] - nxt[0]) if len(nxt) > 1 else np.nan
        lag = dt * (nxt[0] - zr[i]) if len(nxt) else np.nan
        work = lambda f, u: np.trapezoid(f[s] * u[s], dx=dt)
        out.append({'cycle': i + 1,
                    'amplitude_ratio': half(y[s]) / half(y_ref[s]),
                    'frequency_ratio': T / T_p,
                    'phase_deg': 360.0 * lag / T,
                    'y_corr': float(np.corrcoef(y[s], y_ref[s])[0, 1]),
                    'lift_mean_err': (fy[s].mean() - fy_ref[s].mean()) / half(fy_ref[s]),
                    'lift_rms_ratio': fy[s].std() / fy_ref[s].std(),
                    'lift_phase_err': float(np.degrees(dphi(fy[s], y[s]) - dphi(fy_ref[s], y_ref[s]))),
                    'energy_err': (work(fy, v) - work(fy_ref, v_ref)) / np.trapezoid(np.abs(fy_ref[s] * v_ref[s]), dx=dt)})
    return [{k: float(x) if k != 'cycle' else x for k, x in r.items()} for r in out]
