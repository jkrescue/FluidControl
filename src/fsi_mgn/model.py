"""FSIGraphModel: one network, one forward pass per time step; flow and structure are predicted jointly.

    model = FSIGraphModel(config, normalizer)                 # or FSIGraphModel.load(checkpoint)
    for state, out in model.rollout(case, offset, steps):     # same call in training (grad on) and inference (no grad)
        out.stages[-1] -> the prediction (U, p, phi, y, v, a); out.stages[:-1] -> provisional stage predictions

One step (fixed depth; nothing is iterated to convergence and no equation is solved):
  stage 0        : flow = state n, structure = explicit kinematic guess
  stage k = 1..K : physics.evaluate(previous stage) -> feedback features -> message passing (graph U-Net + attention)
                   -> decode flow (cell dU, dp; interior and rear-wall face flux corrections)
                   -> wall force of this flow (explicit integral) -> structure: Newmark equation of motion + correction
                   -> mesh at the new y (known translation map) -> provisional prediction k
  output         : prediction K; its mesh is the next step's geometry; the GRU memory is updated.

config = {'network': {hidden, stages, layers, levels, coarse_layers, heads, attention_layers, memory_dim, memory_warmup},
          'network_dtype': 'float32' | 'float64', 'known_frames': int, 'dt': float}
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from .data import Case, Normalizer, State
from .net import FSIGraphNet
from . import physics as ph

NETWORK_KEYS = ('hidden', 'stages', 'layers', 'levels', 'coarse_layers', 'heads', 'attention_layers', 'memory_dim')


@dataclass(frozen=True)
class Prediction:
    U: Tensor; p: Tensor; phi: Tensor; y: Tensor; v: Tensor; a: Tensor; geom: ph.Geometry; ev: ph.Evaluation


@dataclass(frozen=True)
class StepOutput:
    stages: list           # [Prediction] per coupling stage; stages[-1] is the step's prediction (stage 0 = explicit guess)


class FSIGraphModel(nn.Module):
    def __init__(self, config: dict, normalizer: Normalizer, *, activation_checkpoint: bool = True):
        super().__init__()
        self.config, self.normalizer = dict(config), normalizer
        n = config['network']
        self.net = FSIGraphNet(**{k: n[k] for k in NETWORK_KEYS}, checkpoint=activation_checkpoint)
        self.known_frames, self.dt, self.memory_warmup = int(config['known_frames']), float(config['dt']), int(n['memory_warmup'])

    def set_network_dtype(self, name: str | None = None):
        self.net.to({'float32': torch.float32, 'float64': torch.float64}[name or self.config['network_dtype']]); return self

    # ------------------------------------------------------------------ features
    def _static(self, c: Case, g: ph.Geometry):
        """Geometry/case features of the fluid nodes and fluid edges at the state's mesh."""
        own = c.owner; C, V, S, A = g.C, g.V, g.S, g.A
        centre = g.points[c.labels == 5].mean(0); fcentre = g.points[c.labels == 4].mean(0)
        f = [torch.log(V / V.mean())[:, None], (C - centre) / c.D, (C - fcentre) / c.D,
             torch.log((C - centre).norm(dim=1) / c.D)[:, None], torch.log((C - fcentre).norm(dim=1) / c.D)[:, None],
             V.new_tensor([np.log(c.Re / 100.), c.omega / 0.1, c.U_inf]).expand(len(V), 3)]
        for m in (c.inlet, c.outlet, c.farfield, c.front, c.rear):
            agg = torch.zeros_like(C).index_add_(0, own[m], S[m]); mag = torch.zeros_like(V).index_add_(0, own[m], A[m])
            f.append(agg / (mag[:, None] + 1e-30))
        fi = c.int_faces; d = C[c.neighbor[fi]] - C[own[fi]]; dl = d.norm(dim=1, keepdim=True)
        e = torch.cat((d / dl, torch.log(dl / c.D), S[fi] / A[fi, None], torch.log(A[fi, None] / (c.span * c.D))), 1)
        return torch.cat(f, 1), e, centre

    def _inputs(self, c: Case, st: State, g: ph.Geometry):
        n = self.normalizer; nodeS, edgeS, centre = self._static(c, g); fi = c.int_faces; A = g.A
        node = torch.cat((st.U / c.U_inf, (st.U - st.U1) / n.s1, (st.U1 - st.U2) / n.s1, (st.p / n.sp)[:, None],
                          ((st.p - st.p1) / n.sdp)[:, None], nodeS), 1)
        ed = torch.cat((edgeS, (st.phi[fi] / (c.U_inf * A[fi]))[:, None], ((st.phi[fi] - st.phi1[fi]) / (n.s1 * A[fi]))[:, None],
                        (ph.mesh_flux_now(g, st.v)[fi] / (c.U_inf * A[fi]))[:, None]), 1)
        sign = ed.new_tensor([-1, -1, 1, -1, -1, 1, -1, -1, -1])                 # reverse direction of the same face
        edge = torch.cat((ed, ed * sign), 0)
        m = c.rear_faces; own = c.owner[m]; r = g.fc[m] - centre; nm = g.n[m]
        dn = ((g.C[own] - g.fc[m]) * nm).sum(1).abs()
        Uw = ph.wall_velocity(c, g, st.v)
        wall = torch.cat((nm, torch.log(A[m, None] / (c.span * c.D)), r / c.D, torch.log(dn / c.D)[:, None], Uw / c.U_inf), 1)
        struct = torch.stack((st.y / c.D, st.v / n.sv, st.a / n.sa, st.y1 / c.D, st.v1 / n.sv, st.a1 / n.sa,
                              st.y.new_tensor(np.log(c.mass)), st.y.new_tensor(np.log(c.damping)), st.y.new_tensor(np.log(c.stiffness)),
                              st.y.new_tensor(c.omega / 0.1), st.y.new_tensor(np.log(c.Re / 100.)), st.y.new_tensor(c.U_inf)))[None, :]
        return node, edge, wall, struct

    def _feedback(self, c: Case, st: State, P: Prediction):
        """Explicit physics evaluation of a provisional prediction, as network features (dimensionless, bounded)."""
        n = self.normalizer; g, ev = P.geom, P.ev; fi = c.int_faces; m = c.rear_faces
        node = torch.cat((torch.asinh(ev.cont / ph.cell_capacity(c, g) / 1e-3)[:, None], (g.C - st.points[c.cells].mean(1)) / (self.dt * c.U_inf)), 1)
        gap = torch.asinh(ev.flux_gap / (n.s1 * g.A[fi]))
        edge = torch.cat((gap, -gap), 0)[:, None]
        wall = torch.cat((torch.asinh(ev.wall_flux / (c.U_inf * g.A[m]) / 1e-3)[:, None], ev.slip / c.U_inf,
                          (P.U[c.owner[m]] - st.U[c.owner[m]]) / n.s1), 1)
        struct = torch.stack(((P.y - st.y - self.dt * st.v) / n.sdy, (P.v - st.v - self.dt * st.a) / n.sdv, (P.a - st.a) / n.sda,
                              ev.Fp / n.sF, ev.Fv / n.sF, ev.R_fsi / n.sF, ev.r_y / n.sdy, ev.r_v / n.sdv))[None, :]
        return {'node': node, 'edge': edge, 'wall': wall, 'struct': struct}

    def _predict(self, c: Case, st: State, w: Tensor, U, p, phi_int, phi_wall, y, v, a) -> Prediction:
        pts = ph.translating_points(st.points, w, st.y, y)
        g = ph.geometry(c, st.points, pts, self.dt, w)
        phi = ph.boundary_flux(c, g, U, phi_int, phi_wall)
        return Prediction(U, p, phi, y, v, a, g, ph.evaluate(c, g, U, p, phi, y, v, a, st.y, st.v, st.a, self.dt))

    # ------------------------------------------------------------------ one step = one forward
    def step(self, c: Case, st: State, w: Tensor) -> tuple[State, StepOutput]:
        n, dt, net = self.normalizer, self.dt, self.net; dtype = next(net.parameters()).dtype
        g_n = ph.geometry(c, st.points_prev, st.points, dt, w)
        node, edge, wall, struct = self._inputs(c, st, g_n)
        T = lambda t: t.to(dtype)
        x, e, wl, s = net.encode(T(node), T(edge), T(wall), T(struct))
        mem = st.mem if st.mem is not None else x.new_zeros(len(x), net.memory_dim)
        x = net.mem_inject(x, mem)
        # coarse levels: fixed clusters; positions = mean member cell centre at the state's mesh (explicit geometry)
        levels, pos = [], g_n.C
        for assign, cs, cd, ncl in c.hierarchy(net.levels):
            cnt = torch.zeros(ncl, dtype=pos.dtype, device=pos.device).index_add_(0, assign, torch.ones_like(assign, dtype=pos.dtype))
            pos = torch.zeros(ncl, 2, dtype=pos.dtype, device=pos.device).index_add_(0, assign, pos) / cnt[:, None]
            dd = pos[cd] - pos[cs]
            levels.append((assign, cs, cd, cnt.to(dtype), T(torch.cat((dd / c.D, torch.log(dd.norm(dim=1, keepdim=True) / c.D)), 1)),
                           T((pos - g_n.points[c.labels == 5].mean(0)) / c.D)))
        coarse_e = net.encode_coarse(levels)
        # stage 0: flow = state n, structure = explicit kinematic guess (no learning, no solving)
        y0, v0, a0 = st.y + dt * st.v + .5 * dt * dt * st.a, st.v + dt * st.a, st.a
        g0 = ph.geometry(c, st.points, ph.translating_points(st.points, w, st.y, y0), dt, w)
        P = self._predict(c, st, w, st.U, st.p, ph.interp_flux(c, g0, st.U), ph.mesh_flux_now(g0, v0)[c.rear_faces], y0, v0, a0)
        stages = [P]
        for k in range(net.stages):
            fb = self._feedback(c, st, P)
            x, e, wl, s = net.feedback(x, e, wl, s, {kk: T(vv) for kk, vv in fb.items()})
            x, e, wl, s = net.process(k, x, e, wl, s, c.src, c.dst, c.rear_cells, levels, coarse_e)
            cell, face, wface = net.decode_flow(x, e, wl, s, c.src, c.dst, c.n_int, c.rear_cells)
            cell, face, wface = cell.double(), face.double(), wface.double()
            U = st.U + n.s1 * cell[:, :2]; p = st.p + n.sdp * cell[:, 2]
            # force of THIS stage's flow on the previous stage's wall -> structure head (explicit integral, no solve)
            Fp, Fv, _ = ph.wall_force(c, P.geom, U, p, P.v)
            ds = net.decode_struct(s, T(torch.stack((Fp / n.sF, Fv / n.sF, (Fp + Fv) / n.sF)))).double()
            # Newmark (beta 1/4, gamma 1/2) equation of motion m a + c v + k y = Fy with the force of this stage's flow,
            # solved in closed form for a (one scalar relation), plus the network's correction; y, v follow from Newmark
            m_, c_, k_ = c.mass, c.damping, c.stiffness; Fy_ = Fp + Fv
            a_eq = (Fy_ - c_ * (st.v + .5 * dt * st.a) - k_ * (st.y + dt * st.v + .25 * dt * dt * st.a)) / (m_ + .5 * c_ * dt + .25 * k_ * dt * dt)
            a = a_eq + n.sda * ds[2]
            y = st.y + dt * st.v + .25 * dt * dt * (st.a + a); v = st.v + .5 * dt * (st.a + a)
            gk = ph.geometry(c, st.points, ph.translating_points(st.points, w, st.y, y), dt, w)
            phi_int = ph.interp_flux(c, gk, U) + n.s1 * gk.A[c.int_faces] * face
            phi_wall = ph.mesh_flux_now(gk, v)[c.rear_faces] + c.U_inf * gk.A[c.rear_faces] * 1e-3 * wface
            P = self._predict(c, st, w, U, p, phi_int, phi_wall, y, v, a); stages.append(P)
        mem = net.mem_update(x, mem)                                               # memory for the next step
        new = State(points_prev=st.points, points=P.geom.points, U=P.U, U1=st.U, U2=st.U1, p=P.p, p1=st.p, phi=P.phi, phi1=st.phi,
                    y=P.y, v=P.v, a=P.a, y1=st.y, v1=st.v, a1=st.a, time_s=st.time_s + dt, mem=mem)
        return new, StepOutput(stages)

    def rollout(self, c: Case, offset: int, steps: int, *, burn_in: int = 0, perturb=None, step_checkpoint: bool = False):
        """Autonomous prediction from the known frames ending at offset + known_frames - 1. The first `burn_in` steps run the
        same model without gradient (training then starts from the model's own state); the next `steps` steps are yielded.
        perturb (training only): state -> state, applied once to the state the yielded steps start from.
        step_checkpoint (training only): keep only each step's inputs and outputs for the backward pass and recompute the
        step's internals then, so that the gradient window can span a whole oscillation period."""
        w = c.morph_weights(offset, self.known_frames); st = c.initial_state(offset, self.known_frames)
        st = replace(st, mem=self._warm_memory(c, offset, w))
        for _ in range(burn_in):
            with torch.no_grad(): st, _ = self.step(c, st, w)
        if perturb is not None: st = perturb(st)
        step = self.step
        if step_checkpoint and self.training and torch.is_grad_enabled():
            step = lambda c_, st_, w_: torch.utils.checkpoint.checkpoint(self.step, c_, st_, w_, use_reentrant=False)
        for _ in range(steps):
            st, out = step(c, st, w); yield st, out

    def _warm_memory(self, c: Case, offset: int, w: Tensor):
        """Fill the temporal memory from the last `memory_warmup` known frames (anchor and before): encoder + GRU only."""
        net = self.net; dtype = next(net.parameters()).dtype; a = offset + self.known_frames - 1
        M = self.memory_warmup; mem = None
        for j in range(max(2, a - M + 1), a + 1):
            st_j = c.initial_state(j - self.known_frames + 1, self.known_frames)
            node = self._inputs(c, st_j, ph.geometry(c, st_j.points_prev, st_j.points, self.dt, w))[0].to(dtype)
            x = net.enc_node(node)
            if mem is None: mem = x.new_zeros(len(x), net.memory_dim)
            mem = net.mem_update(net.mem_inject(x, mem), mem)
        return mem

    def bundle(self, **extra) -> dict:
        return dict(model=self.state_dict(), config=self.config, normalizer=self.normalizer.as_dict(), **extra)

    @classmethod
    def load(cls, path, device='cuda', *, activation_checkpoint: bool = False):
        saved = torch.load(path, map_location=device, weights_only=False)
        m = cls(saved['config'], Normalizer(**saved['normalizer']), activation_checkpoint=activation_checkpoint).to(device)
        m.load_state_dict(saved['model'], strict=True); m.set_network_dtype()
        return m, saved
