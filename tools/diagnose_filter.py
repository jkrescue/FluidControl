"""Causal check (diagnosis only, not a model): autonomous rollout where the state's grid-scale part is damped after every step,
U <- U - beta (U - neighbour mean U), same for p. If amplitude / phase improve, grid-scale growth is a cause of their errors.

  python diagnose_filter.py <checkpoint.pt> <views_dir> <split.json> <out.json> <role>[,<role>...] <beta>[,<beta>...] [steps=300]
"""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

from fsi_mgn.model import FSIGraphModel
from fsi_mgn.data import Case
from fsi_mgn import physics as ph
from fsi_mgn.metrics import viv_cycles


def main(ck, views, split, out, roles, betas, steps=300):
    roles = roles.split(','); betas = [float(b) for b in betas.split(',')]
    if 'HELDOUT' in roles: raise SystemExit('HELDOUT is for the final acceptance only')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model, _ = FSIGraphModel.load(ck, 'cuda'); model.eval(); NK = model.known_frames
    sp = json.loads(Path(split).read_text()); res = []
    for role in roles:
        for cd in sp['roles'][role]:
            c = Case(Path(views) / cd['name'], cd['sequence_sha256'], cd['report_sha256'], 'cuda')
            n = min(int(steps), len(c.times) - NK); a0 = NK - 1; w = c.morph_weights(0, NK)
            F = np.array(c.z('native_force_components')); Y = np.array(c.z('native_y_absolute'))
            for beta in betas:
                st = c.initial_state(0, NK); st = replace(st, mem=model._warm_memory(c, 0, w)); y, fy = [], []
                with torch.no_grad():
                    for _ in range(n):
                        st, out_ = model.step(c, st, w)
                        if beta > 0:
                            st = replace(st, U=st.U - beta * (st.U - ph.neighbour_mean(c, st.U)),
                                         p=st.p - beta * (st.p - ph.neighbour_mean(c, st.p[:, None])[:, 0]))
                        y.append(float(st.y)); fy.append(float(out_.stages[-1].ev.Fy))
                cyc = viv_cycles(np.array(y), np.array(fy), Y[a0 + 1:a0 + 1 + n], F[a0 + 1:a0 + 1 + n, 0], model.dt, y_mean=float(Y.mean()))
                res.append({'case': c.name, 'beta': beta, 'y': y, 'cycles': cyc}); print(c.name, beta, flush=True)
    Path(out).write_text(json.dumps(res))


if __name__ == '__main__':
    main(*sys.argv[1:7], *sys.argv[7:8])
