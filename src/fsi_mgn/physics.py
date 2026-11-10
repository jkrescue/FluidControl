"""Explicit physics EVALUATIONS on predicted quantities (no equation is solved, nothing is iterated).

Used in two places, identically in training and inference:
  (1) inside the network, between coupling stages: the provisional prediction of stage k is evaluated here and the values
      are fed back as features to stage k+1 (fixed number of stages);
  (2) in the training loss (fsi_mgn_train.py).
All formulas are closed-form sums over faces/cells of the native mesh (FP64).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Geometry:
    points: Tensor; S: Tensor; A: Tensor; n: Tensor; fc: Tensor; C: Tensor; V: Tensor
    mesh_flux: Tensor; rate: Tensor; dt: float
    w_face: Tensor        # mean morph weight of each face's end points (mesh point velocity = w * v_structure)


def translating_points(points_n: Tensor, w: Tensor, y_n: Tensor, y_new: Tensor) -> Tensor:
    """Known fixed translation map: every mesh point moves by w * (y_new - y_n) in y (a geometric formula)."""
    return points_n + w[:, None] * torch.stack((y_new * 0, y_new - y_n))[None, :]


def divergence(c, face_values: Tensor) -> Tensor:
    """Owner-positive sum of a face quantity per cell."""
    r = torch.zeros(c.nc, dtype=face_values.dtype, device=face_values.device).index_add(0, c.owner, face_values)
    return r.index_add(0, c.neighbor[c.int_faces], -face_values[c.int_faces])


def geometry(c, points_old: Tensor, points: Tensor, dt: float, w: Tensor) -> Geometry:
    def build(x):
        e = x[c.pairs[:, 1]] - x[c.pairs[:, 0]]
        S = c.span * c.orientation[:, None] * torch.stack((e[:, 1], -e[:, 0]), 1)
        fc = x[c.pairs].mean(1)
        return S, fc, divergence(c, (S * fc).sum(1) / 2), x[c.cells].mean(1)
    S0, _, V0, _ = build(points_old); S, fc, V, C = build(points)
    A = S.norm(dim=1)
    if not bool((V > 0).all() & (V0 > 0).all() & (A > 0).all()): raise ValueError('Invalid cell geometry')
    mesh = (.5 * (S + S0) * ((points - points_old) / dt)[c.pairs].mean(1)).sum(1)
    return Geometry(points, S, A, S / A[:, None], fc, C, V, mesh, (V - V0) / dt, dt, w[c.pairs].mean(1))


def mesh_flux_now(g: Geometry, v: Tensor) -> Tensor:
    """Instantaneous mesh flux S . (0, w v): the mesh moves as w * y(t), so its point velocity is w * v (w = 1 on the rear
    wall, where this is the wall's normal velocity flux). Continuity of the ABSOLUTE flux needs no mesh term: sum_f phi = 0
    (CFD truth: 2e-10 of U_inf |S|); the relative flux phi - mesh has divergence -dV/dt (GCL)."""
    return g.S[:, 1] * g.w_face * v


def wall_velocity(c, g: Geometry, v: Tensor) -> Tensor:
    """Rigid rear-wall velocity (translation v in y + prescribed spin) at every rear face [n_rear, 2]."""
    centre = g.points[c.labels == 5].mean(0); r = g.fc[c.rear_faces] - centre
    return torch.stack((-c.omega * r[:, 1], v + c.omega * r[:, 0]), 1)


def boundary_flux(c, g: Geometry, U: Tensor, interior: Tensor, rear: Tensor) -> Tensor:
    """Full absolute face flux from the network's interior and rear-wall face outputs + boundary conditions (inputs):
    inlet U_inf S_x, front wall 0 (fixed), outlet/farfield owner value."""
    phi = (U[c.owner] * g.S).sum(1)                                            # outlet / farfield: owner value
    phi = torch.where(c.inlet, c.U_inf * g.S[:, 0], phi)
    phi = torch.where(c.front, torch.zeros_like(phi), phi)
    phi = phi.index_put((c.int_faces,), interior)
    return phi.index_put((c.rear_faces,), rear)


def interp_flux(c, g: Geometry, U: Tensor) -> Tensor:
    """Linear interpolation of cell velocity to interior faces, dotted with S (interior faces only)."""
    fi = c.int_faces
    return (.5 * (U[c.owner[fi]] + U[c.neighbor[fi]]) * g.S[fi]).sum(1)


@dataclass(frozen=True)
class Evaluation:
    Fp: Tensor; Fv: Tensor; Fy: Tensor        # rear-cylinder transverse force [N]
    R_fsi: Tensor                              # m a + c v + k y - Fy
    r_y: Tensor; r_v: Tensor                   # Newmark kinematic residuals of (y, v, a) w.r.t. step n
    cont: Tensor                               # per cell: sum_f phi (absolute flux of an incompressible flow is solenoidal) [m^3/s]
    wall_flux: Tensor                          # per rear face: phi - S . (0, v) (no penetration: fluid moves with the wall)
    slip: Tensor                               # per rear face: (U_owner - U_wall) [2]
    flux_gap: Tensor                           # per interior face: phi - interp(U).S


def wall_force(c, g: Geometry, U: Tensor, p: Tensor, v: Tensor):
    m = c.rear_faces; own = c.owner[m]
    Fp = c.rho * (p[own] * g.S[m, 1]).sum()
    Uw = wall_velocity(c, g, v); nm = g.n[m]; du = U[own] - Uw; du = du - (du * nm).sum(1, keepdim=True) * nm
    dn = ((g.C[own] - g.fc[m]) * nm).sum(1).abs()
    Fv = c.rho * c.nu * (du[:, 1] * g.A[m] / dn).sum()
    return Fp, Fv, Uw


def evaluate(c, g: Geometry, U: Tensor, p: Tensor, phi: Tensor, y: Tensor, v: Tensor, a: Tensor,
             y_n: Tensor, v_n: Tensor, a_n: Tensor, dt: float) -> Evaluation:
    Fp, Fv, Uw = wall_force(c, g, U, p, v); Fy = Fp + Fv
    R = c.mass * a + c.damping * v + c.stiffness * y - Fy
    r_y = y - (y_n + dt * v_n + .25 * dt * dt * (a_n + a)); r_v = v - (v_n + .5 * dt * (a_n + a))
    m = c.rear_faces
    return Evaluation(Fp, Fv, Fy, R, r_y, r_v, divergence(c, phi), phi[m] - mesh_flux_now(g, v)[m], U[c.owner[m]] - Uw,
                      phi[c.int_faces] - interp_flux(c, g, U))


def cell_capacity(c, g: Geometry) -> Tensor:
    """U_inf * sum_f |S| per cell: the scale of a cell's face-flux balance (continuity residuals are reported relative to it)."""
    fi = c.int_faces
    return torch.zeros_like(g.V).index_add_(0, c.owner, g.A).index_add_(0, c.neighbor[fi], g.A[fi]) * c.U_inf


def vorticity(c, g: Geometry, U: Tensor) -> Tensor:
    """Green-Gauss cell vorticity (1/V) sum_f (S x U_f)_z, U_f linear interior / owner value on boundaries."""
    nei_ = torch.where(c.inner, c.neighbor, c.owner)
    Uf = torch.where(c.inner[:, None], .5 * (U[c.owner] + U[nei_]), U[c.owner])
    return divergence(c, g.S[:, 0] * Uf[:, 1] - g.S[:, 1] * Uf[:, 0]) / g.V


def neighbour_mean(c, x: Tensor) -> Tensor:
    deg = torch.zeros(c.nc, dtype=x.dtype, device=x.device).index_add_(0, c.dst, torch.ones_like(c.dst, dtype=x.dtype))
    return torch.zeros_like(x).index_add_(0, c.dst, x[c.src]) / deg[:, None]
