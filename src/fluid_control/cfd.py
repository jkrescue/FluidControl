"""Two-dimensional incompressible spectral CFD with Brinkman rotating cylinders.

This is a deliberately small, inspectable CFD teacher for a PhysicsNeMo
surrogate experiment. It is not a high-fidelity reproduction of Yeung's water
tunnel: the spanwise direction is omitted and the first-pass surface gap is
0.125 D so it is resolved by several cells at the default grid size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CFDConfig:
    nx: int = 256
    ny: int = 128
    lx: float = 12.0
    ly: float = 6.0
    re: float = 100.0
    dt: float = 0.005
    steps: int = 3600
    average_start: int = 2000
    gap: float = 0.125
    eta: float = 0.02
    control_radius: float = 0.125
    main_radius: float = 0.5
    main_x: float = 3.0

    def metadata(self) -> dict:
        return asdict(self)


class CylinderCFD:
    def __init__(self, cfg: CFDConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this experiment")
        ny, nx = cfg.ny, cfg.nx
        x = torch.arange(nx, device=self.device, dtype=torch.float32) * (cfg.lx / nx)
        y = (torch.arange(ny, device=self.device, dtype=torch.float32) + 0.5) * (cfg.ly / ny) - cfg.ly / 2
        self.x, self.y = torch.meshgrid(x, y, indexing="xy")
        dx, dy = cfg.lx / nx, cfg.ly / ny
        self.dx, self.dy = dx, dy
        kx = 2 * np.pi * torch.fft.fftfreq(nx, d=dx, device=self.device)
        ky = 2 * np.pi * torch.fft.fftfreq(ny, d=dy, device=self.device)
        self.kx = kx[None, None, :]
        self.ky = ky[None, :, None]
        self.k2 = self.kx.square() + self.ky.square()
        self.safe_k2 = torch.where(self.k2 == 0, 1.0, self.k2)
        self.viscous = 1.0 / (1.0 + cfg.dt * self.k2 / cfg.re)
        self.dealias = ((torch.fft.fftfreq(nx, device=self.device).abs()[None, None, :] <= 1 / 3)
                        & (torch.fft.fftfreq(ny, device=self.device).abs()[None, :, None] <= 1 / 3))

        center_distance = cfg.main_radius + cfg.control_radius + cfg.gap
        offset = center_distance / np.sqrt(2)
        self.centers = [(cfg.main_x + offset, offset), (cfg.main_x + offset, -offset)]
        smooth = min(dx, dy) * 0.7

        def disk(cx: float, cy: float, radius: float) -> torch.Tensor:
            radius_field = torch.sqrt((self.x - cx).square() + (self.y - cy).square() + 1e-12)
            return torch.sigmoid((radius - radius_field) / smooth)

        self.main_mask = disk(cfg.main_x, 0.0, cfg.main_radius)
        self.control_masks = torch.stack([disk(cx, cy, cfg.control_radius) for cx, cy in self.centers])
        self.mask = torch.clamp(self.main_mask + self.control_masks.sum(0), 0.0, 1.0)
        self.sponge = 2.5 * (
            torch.sigmoid((0.8 - self.x) / 0.2)
            + torch.sigmoid((self.x - (cfg.lx - 1.5)) / 0.2)
        )
        self.wake_mask = (self.x >= cfg.main_x + 1.5) & (self.x <= cfg.main_x + 5.0) & (self.y.abs() <= 1.5)

    def feature_maps(self, actions: torch.Tensor, out_size: tuple[int, int] | None = None) -> torch.Tensor:
        actions = actions.to(self.device, dtype=torch.float32)
        batch = actions.shape[0]
        body_u, body_v = self.body_velocity(actions)
        static = torch.stack([self.main_mask, self.control_masks[0], self.control_masks[1]])
        static = static.unsqueeze(0).expand(batch, -1, -1, -1)
        reynolds = torch.full_like(body_u, self.cfg.re / 100.0)
        features = torch.cat([static, body_u[:, None], body_v[:, None], reynolds[:, None]], dim=1)
        if out_size is not None:
            features = F.interpolate(features, size=out_size, mode="area")
        return features

    def body_velocity(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = actions.shape[0]
        ub = torch.zeros((batch, self.cfg.ny, self.cfg.nx), device=self.device)
        vb = torch.zeros_like(ub)
        for idx, (cx, cy) in enumerate(self.centers):
            omega = actions[:, idx, None, None] / self.cfg.control_radius
            ub = ub - omega * (self.y - cy) * self.control_masks[idx]
            vb = vb + omega * (self.x - cx) * self.control_masks[idx]
        return ub, vb

    @torch.no_grad()
    def run(self, actions: np.ndarray | torch.Tensor, progress: bool = False) -> dict[str, np.ndarray]:
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.ndim == 1:
            actions = actions[None]
        batch = actions.shape[0]
        u = torch.ones((batch, self.cfg.ny, self.cfg.nx), device=self.device)
        v = torch.zeros_like(u)
        # A tiny fixed perturbation initiates the unstable wake in symmetric cases.
        v = v + 1e-3 * torch.sin(2 * np.pi * self.y / self.cfg.ly)[None] * torch.exp(-((self.x - self.cfg.main_x) / 1.2).square())[None]
        ub, vb = self.body_velocity(actions)
        alpha = self.cfg.dt * self.mask[None] / self.cfg.eta
        beta = self.cfg.dt * self.sponge[None]
        denominator = 1.0 + alpha + beta
        viscous = self.viscous
        wake_sum = torch.zeros(batch, device=self.device)
        mean_u = torch.zeros_like(u)
        mean_v = torch.zeros_like(v)
        mean_energy = torch.zeros_like(u)
        samples = 0

        for step in range(self.cfg.steps):
            uh, vh = torch.fft.fft2(u), torch.fft.fft2(v)
            dudx = torch.fft.ifft2(1j * self.kx * uh).real
            dudy = torch.fft.ifft2(1j * self.ky * uh).real
            dvdx = torch.fft.ifft2(1j * self.kx * vh).real
            dvdy = torch.fft.ifft2(1j * self.ky * vh).real
            us = u - self.cfg.dt * (u * dudx + v * dudy)
            vs = v - self.cfg.dt * (u * dvdx + v * dvdy)
            us = (us + alpha * ub + beta) / denominator
            vs = (vs + alpha * vb) / denominator
            ush, vsh = torch.fft.fft2(us), torch.fft.fft2(vs)
            divergence = self.kx * ush + self.ky * vsh
            ush = (ush - self.kx * divergence / self.safe_k2) * viscous * self.dealias
            vsh = (vsh - self.ky * divergence / self.safe_k2) * viscous * self.dealias
            u = torch.fft.ifft2(ush).real
            v = torch.fft.ifft2(vsh).real
            if not torch.isfinite(u).all() or not torch.isfinite(v).all():
                raise FloatingPointError(f"CFD diverged at step {step}")
            if step >= self.cfg.average_start:
                mean_u += u
                mean_v += v
                energy = (u - 1.0).square() + v.square()
                mean_energy += energy
                wake_sum += energy[:, self.wake_mask].mean(dim=1)
                samples += 1
            if progress and (step + 1) % 200 == 0:
                print(f"CFD step {step + 1}/{self.cfg.steps}", flush=True)

        mean_u /= samples
        mean_v /= samples
        mean_energy /= samples
        wake_error = wake_sum / samples
        control_cost = 0.01 * actions.square().sum(dim=1)
        objective = wake_error + control_cost
        fields = torch.stack([mean_u, mean_v, mean_energy], dim=1)
        return {
            "actions": actions.cpu().numpy(),
            "fields": fields.cpu().numpy(),
            "wake_error": wake_error.cpu().numpy(),
            "control_cost": control_cost.cpu().numpy(),
            "objective": objective.cpu().numpy(),
        }
