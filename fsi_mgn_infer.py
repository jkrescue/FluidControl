"""Inference with a released FSIGraphNet checkpoint: autonomous strong-FSI prediction, one forward pass per step.

  python fsi_mgn_infer.py --checkpoint checkpoints/fsi_graphnet.pt --case <case dir: sequence.npz + report.json> \
                          [--offset 0] [--steps 300] [--out out/] [--fields] [--device cuda]

Only the known frames offset .. offset+known_frames-1 of the case are read (initial state); no later CFD data is used.
Writes <out>/trajectory.json (per step: time, y, v, a, Fy, Fp, Fv, R_fsi, step wall time) and, with --fields,
<out>/fields.npz (cell U, p, vorticity and mesh points per step). Run with PYTHONPATH=src.
"""
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # deterministic cuBLAS: bitwise equal to training/evaluation
import numpy as np
import torch

from fsi_mgn import physics as ph
from fsi_mgn.array_cache import file_sha256
from fsi_mgn.data import Case
from fsi_mgn.model import FSIGraphModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--case', required=True)
    ap.add_argument('--offset', type=int, default=0); ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--out', default='out'); ap.add_argument('--fields', action='store_true'); ap.add_argument('--device', default='cuda')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); case = Path(a.case)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    model, saved = FSIGraphModel.load(a.checkpoint, a.device); model.eval()
    c = Case(case, file_sha256(case / 'sequence.npz'), file_sha256(case / 'report.json'), a.device)
    steps = min(a.steps, len(c.times) - (a.offset + model.known_frames))
    rows, U, P, W, X = [], [], [], [], []
    with torch.no_grad():
        it = model.rollout(c, a.offset, steps)
        for i in range(1, steps + 1):
            t0 = time.monotonic(); st, o = next(it)
            if a.device == 'cuda': torch.cuda.synchronize()
            ev = o.stages[-1].ev
            rows.append({'step': i, 'time_s': st.time_s, 'y': float(st.y), 'v': float(st.v), 'a': float(st.a), 'Fy': float(ev.Fy),
                         'Fp': float(ev.Fp), 'Fv': float(ev.Fv), 'R_fsi': float(ev.R_fsi), 'step_wall_s': time.monotonic() - t0})
            if a.fields:
                U.append(st.U.float().cpu().numpy()); P.append(st.p.float().cpu().numpy()); X.append(st.points.float().cpu().numpy())
                W.append(ph.vorticity(c, o.stages[-1].geom, st.U).float().cpu().numpy())
    (out / 'trajectory.json').write_text(json.dumps({'checkpoint': a.checkpoint, 'case': c.name, 'offset': a.offset,
                                                      'known_frames': model.known_frames, 'rows': rows}, indent=1))
    if a.fields: np.savez(out / 'fields.npz', U=np.array(U), p=np.array(P), w=np.array(W), xy=np.array(X))
    print('%s: %d steps, %.3f s/step -> %s' % (c.name, steps, np.mean([r['step_wall_s'] for r in rows[1:]] or [0]), out))


if __name__ == '__main__':
    main()
