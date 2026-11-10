"""Case data: mesh topology, boundary patches, physical constants and read-only native arrays. Pure data access."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class State:
    """Everything the network sees at time n (all FP64 on the device)."""
    points_prev: Tensor   # mesh points at n-1 (mesh flux at n)
    points: Tensor        # mesh points at n
    U: Tensor             # cell velocity n      [nc, 2]
    U1: Tensor            # n-1
    U2: Tensor            # n-2
    p: Tensor             # cell kinematic pressure n
    p1: Tensor            # n-1
    phi: Tensor           # absolute face flux n  [nf]
    phi1: Tensor          # n-1
    y: Tensor             # rear-cylinder transverse displacement n (0-d)
    v: Tensor
    a: Tensor
    y1: Tensor            # n-1
    v1: Tensor
    a1: Tensor
    time_s: float
    mem: object = None    # latent temporal memory [nc, dim] (network dtype), carried by the model from step to step


class Case:
    def __init__(self, folder, sequence_sha256: str, report_sha256: str, device='cuda'):
        from .array_cache import case_array, file_sha256
        folder = Path(folder); self.folder, self.name, self.sha, self.dev = folder, folder.name, sequence_sha256, device
        if file_sha256(folder / 'sequence.npz') != sequence_sha256 or file_sha256(folder / 'report.json') != report_sha256:
            raise ValueError('Data pin changed: ' + self.name)
        rep = json.loads((folder / 'report.json').read_text()); self.report = rep
        if not (rep.get('rotation_enabled') is True and rep.get('mesh_rotation') is False and rep.get('angular_structure_dof') is False):
            raise ValueError('Prescribed-spin, non-rotating-mesh, transverse-only case required')
        self.U_inf = float(rep.get('U_inf', 1.0)); self.nu = float(rep['nu']); self.rho = float(rep['rho']); self.span = float(rep['span'])
        self.D = float(rep['D_rear']); self.omega = float(rep['omega']); self.Re = self.U_inf * self.D / self.nu
        self.mass, self.damping, self.stiffness = float(rep['mass']), float(rep['damping']), float(rep['stiffness'])
        self.frames = rep['frames']
        self.z = lambda k: case_array(folder, k, sequence_sha256)
        self.times = np.array(self.z('times'))
        L = lambda x: torch.as_tensor(np.array(x), dtype=torch.long, device=device)
        own = np.array(self.z('planar_face_owner')); nei = np.array(self.z('planar_face_neighbor'))
        self.pairs = L(self.z('face_endpoint_node_ids'))
        self.orientation = torch.as_tensor(np.array(self.z('orientation')), dtype=torch.float64, device=device)
        self.cells = L(self.z('cell_vertices_ccw')); self.labels = L(np.array(self.z('node_types')).reshape(-1))
        self.owner, self.neighbor = L(own), L(nei)
        self.nc = int(max(own.max(), nei.max()) + 1); self.nf = len(own)
        names = list(np.array(self.z('boundary_patch_names'))); start = np.array(self.z('boundary_patch_start'))
        count = np.array(self.z('boundary_patch_count')); fidx = np.array(self.z('face_indices'))
        patch = lambda n: torch.as_tensor((fidx >= start[names.index(n)]) & (fidx < start[names.index(n)] + count[names.index(n)]), device=device)
        self.inlet, self.outlet, self.farfield = patch('inlet'), patch('outlet'), patch('farfield')
        self.front, self.rear = patch('frontCylinder'), patch('rearCylinder')
        self.inner = torch.as_tensor(nei >= 0, device=device)
        if not torch.equal(self.inlet | self.outlet | self.farfield | self.front | self.rear, ~self.inner): raise ValueError('Patch cover')
        fi = np.flatnonzero(nei >= 0); self.int_faces = torch.as_tensor(fi, device=device); self.n_int = len(fi)
        o, n = torch.as_tensor(own[fi], device=device), torch.as_tensor(nei[fi], device=device)
        self.src, self.dst = torch.cat((o, n)), torch.cat((n, o))           # fluid graph edges (both directions)
        self.rear_faces = torch.nonzero(self.rear)[:, 0]; self.rear_cells = self.owner[self.rear_faces]   # wall-structure edges
        self.bnd_faces = torch.nonzero(~self.inner)[:, 0]
        self._hier = {}

    def hierarchy(self, levels: int):
        """Multi-scale graph (cached): built once from the case's first mesh (frame 0, always a known frame)."""
        if levels not in self._hier:
            xy0 = np.array(self.z('node_xy')[0]); C0 = xy0[np.array(self.z('cell_vertices_ccw'))].mean(1)
            H = build_hierarchy(C0, self.src.cpu().numpy(), self.dst.cpu().numpy(), levels)
            L = lambda x: torch.as_tensor(x, dtype=torch.long, device=self.dev)
            self._hier[levels] = [(L(a), L(s), L(d), n) for a, s, d, n in H]
        return self._hier[levels]

    # ------------------------------------------------------------------ known data only (frames <= anchor)
    def body(self, k: int):
        fr = self.frames[k]; T = lambda x: torch.tensor(float(x), dtype=torch.float64, device=self.dev)
        return T(fr['centreOfRotation'][1]), T(fr['velocity'][1]), T(fr['acceleration'][1])

    def initial_state(self, offset: int, known_frames: int) -> State:
        a = offset + known_frames - 1
        if a < 2 or a + 1 >= len(self.times): raise ValueError('Window outside the sequence')
        D_ = lambda x: torch.as_tensor(np.array(x), dtype=torch.float64, device=self.dev)
        cu, cp, ph, xy = self.z('native_cell_U'), self.z('native_cell_p'), self.z('absolute_flux'), self.z('node_xy')
        y, v, acc = self.body(a); y1, v1, a1 = self.body(a - 1)
        return State(points_prev=D_(xy[a - 1]), points=D_(xy[a]), U=D_(cu[a][:, :2]), U1=D_(cu[a - 1][:, :2]), U2=D_(cu[a - 2][:, :2]),
                     p=D_(cp[a]), p1=D_(cp[a - 1]), phi=D_(ph[a]), phi1=D_(ph[a - 1]), y=y, v=v, a=acc, y1=y1, v1=v1, a1=a1,
                     time_s=float(self.times[a]))

    def morph_weights(self, offset: int, known_frames: int) -> Tensor:
        """Mesh-point weights of the known translation-only mesh motion, identified from the last two KNOWN meshes."""
        a = offset + known_frames - 1; xy = self.z('node_xy')
        dy = float(self.frames[a]['centreOfRotation'][1]) - float(self.frames[a - 1]['centreOfRotation'][1])
        change = np.array(xy[a]) - np.array(xy[a - 1])
        if np.max(np.abs(change[:, 0])) > 1e-7 or abs(dy) < 1e-6: raise ValueError('Morph map not identifiable from known frames')
        w = change[:, 1] / dy
        if w.min() < -1e-5 or w.max() > 1 + 1e-5: raise ValueError('Morph map out of bounds')
        return torch.as_tensor(w, dtype=torch.float64, device=self.dev)


@dataclass(frozen=True)
class Normalizer:
    """TRAIN-only scales (saved with the checkpoint)."""
    s1: float; sp: float; sdp: float           # cell U one-step change, p fluctuation, p one-step change
    sy: float; sv: float; sa: float; sF: float  # structure / force fluctuation
    sdy: float; sdv: float; sda: float          # residual of y, v, a w.r.t. the explicit kinematic guess

    def as_dict(self): return dict(self.__dict__)


def compute_normalizer(cases, dt: float = 0.1) -> Normalizer:
    a1 = ap = adp = 0.0; Y, V, A, F, dY, dV, dA = [], [], [], [], [], [], []
    for c in cases:
        U = np.array(c.z('native_cell_U'))[..., :2]; p = np.array(c.z('native_cell_p'))
        a1 += float((np.diff(U, axis=0) ** 2).sum(-1).mean()); ap += float(((p - p.mean(0)) ** 2).mean())
        adp += float((np.diff(p, axis=0) ** 2).mean())
        y = np.array([f['centreOfRotation'][1] for f in c.frames]); v = np.array([f['velocity'][1] for f in c.frames])
        acc = np.array([f['acceleration'][1] for f in c.frames]); Y += list(y); V += list(v); A += list(acc)
        F += list(np.array(c.z('native_force_components'))[:, 0])
        dY += list(y[1:] - y[:-1] - dt * v[:-1] - .5 * dt * dt * acc[:-1]); dV += list(v[1:] - v[:-1] - dt * acc[:-1]); dA += list(acc[1:] - acc[:-1])
    n = len(cases); r = lambda x: float(np.sqrt(np.mean(np.square(x))))
    return Normalizer(s1=float(np.sqrt(a1 / n)), sp=float(np.sqrt(ap / n)), sdp=float(np.sqrt(adp / n)),
                      sy=float(np.std(Y)), sv=float(np.std(V)), sa=float(np.std(A)), sF=float(np.std(F)), sdy=r(dY), sdv=r(dV), sda=r(dA))


def build_hierarchy(C0: np.ndarray, src: np.ndarray, dst: np.ndarray, levels: int, rounds: int = 2):
    """Coarse graph levels by deterministic greedy pairwise matching (topology only, no solving).

    C0: [n, 2] cell centres of the case's FIRST (known) mesh; src/dst: directed fine edges. Each level applies `rounds`
    matchings (each pairs every unmatched node with its nearest unmatched neighbour, in a fixed node order), so a level has
    ~n / 2^rounds nodes. Returns [(assign [n_prev] -> cluster id, src_l, dst_l, n_l)] for levels 1..levels."""
    out = []; C = C0.astype(np.float64); s, d = src.astype(np.int64), dst.astype(np.int64); n = len(C)
    for _ in range(levels):
        assign_total = np.arange(n)
        for _ in range(rounds):
            nb = [[] for _ in range(n)]
            for a, b in zip(s, d): nb[a].append(b)
            match = -np.ones(n, np.int64); cid = 0; amap = np.empty(n, np.int64)
            for i in np.lexsort((C[:, 1], C[:, 0])):                          # fixed order: by x, then y
                if match[i] >= 0: continue
                cand = [j for j in nb[i] if match[j] < 0 and j != i]
                if cand:
                    j = min(cand, key=lambda j: ((C[j] - C[i]) ** 2).sum()); match[i] = match[j] = cid; amap[i] = amap[j] = cid
                else:
                    match[i] = cid; amap[i] = cid
                cid += 1
            cnt = np.bincount(amap, minlength=cid).astype(np.float64)
            C = np.stack([np.bincount(amap, C[:, k], cid) / cnt for k in range(2)], 1)
            e = np.unique(np.stack((amap[s], amap[d]), 1), axis=0); e = e[e[:, 0] != e[:, 1]]
            s, d = e[:, 0], e[:, 1]; assign_total = amap[assign_total]; n = cid
        out.append((assign_total, s.copy(), d.copy(), n))
    return out
