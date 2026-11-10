"""Smoke test: build the model from configs/fsi_graphnet.json, one H-step training forward + backward, train == infer
bitwise, timing, parameter count.

  PYTHONPATH=src python -B tests/smoke.py '<case json>' <offset> <H>      (cases under /views)
"""
import json, pathlib, sys, time
import torch
from fsi_mgn.data import Case, compute_normalizer
from fsi_mgn.model import FSIGraphModel
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
cd, off, H = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
c = Case('/views/' + cd['name'], cd['sequence_sha256'], cd['report_sha256'], 'cuda')
cfg = json.loads((pathlib.Path(__file__).resolve().parents[1] / 'configs/fsi_graphnet.json').read_text())['model']
print('network', cfg['network'])
norm = compute_normalizer([c]); print('normalizer', {k: '%.3e' % v for k, v in norm.as_dict().items()})
torch.manual_seed(0); m = FSIGraphModel(cfg, norm).cuda(); m.set_network_dtype(); m.train()
with torch.no_grad():
    for p in m.parameters(): p.add_(1e-2 * torch.randn_like(p))     # move off the zero-initialised heads
print('parameters %d' % sum(p.numel() for p in m.parameters()))
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.monotonic()
outs = list(m.rollout(c, off, H))
loss = sum(st.U.square().mean() + st.p.square().mean() + out.stages[-1].ev.R_fsi ** 2 for st, out in outs); loss.backward()
torch.cuda.synchronize(); print('train window H=%d fwd+bwd %.2f s, peak %.2f GiB' % (H, time.monotonic() - t0, torch.cuda.max_memory_allocated() / 2 ** 30))
gn = sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None) ** .5
zero = [k for k, p in m.named_parameters() if p.grad is None or not bool(p.grad.abs().sum() > 0)]
print('grad norm %.3e, params with zero grad: %s' % (gn, zero or 'none'))
m.eval(); t0 = time.monotonic()
with torch.no_grad(): inf = list(m.rollout(c, off, H))
torch.cuda.synchronize(); print('inference %d steps %.3f s (%.3f s/step)' % (H, time.monotonic() - t0, (time.monotonic() - t0) / H))
worst = max(max(float((a.detach() - b).abs().max()) for a, b in ((s1.U, s2.U), (s1.p, s2.p), (s1.phi, s2.phi), (s1.y, s2.y), (s1.v, s2.v), (s1.a, s2.a)))
            for (s1, _), (s2, _) in zip(outs, inf))
print('train vs infer max |diff| over %d steps: %.3e' % (H, worst))
for i, (st, out) in enumerate(inf, 1):
    print('step %d  stage R_fsi:' % i, ['%.3e' % float(P.ev.R_fsi) for P in out.stages], ' y=%.5f' % float(st.y))
