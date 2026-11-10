"""FSIGraphNet: a MeshGraphNet with a fluid graph and a structure node, coupled through wall-structure edges.

Graph: fluid nodes = mesh cells; fluid edges = interior faces (both directions); one structure node (rear cylinder);
wall edges = rear-wall faces, each linking its wall cell and the structure node (both directions).
Processor: `stages` coupling stages. Each stage runs `layers` fine message-passing blocks around a graph U-Net (coarse
levels from fixed cell clusters, global self-attention on the coarsest level). Every fine block updates fluid edges,
wall edges, fluid nodes (including messages from the structure) and the structure node (including messages from the
near-wall flow). Between stages the model decodes a provisional prediction, evaluates it with explicit physics and feeds
the values back through the `fb_*` encoders. A per-cell GRU memory carries information from step to step.
Decoders are shared by all stages.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def mlp(n_in: int, h: int, n_out: int, layers: int = 2, norm: bool = True) -> nn.Sequential:
    mods, d = [], n_in
    for _ in range(layers): mods += [nn.Linear(d, h), nn.SiLU()]; d = h
    mods.append(nn.Linear(d, n_out))
    if norm: mods.append(nn.LayerNorm(n_out))
    return nn.Sequential(*mods)


def zero_last(m: nn.Sequential):
    lin = [x for x in m if isinstance(x, nn.Linear)][-1]; nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)


class Block(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.edge = mlp(3 * h, h, h); self.wall = mlp(3 * h, h, h)
        self.node = mlp(3 * h, h, h); self.struct = mlp(3 * h, h, h)

    def forward(self, x, e, w, s, src, dst, wc):
        """x [nc,h] fluid nodes; e [2 n_int,h] fluid edges; w [n_wall,h] wall edges; s [1,h] structure node; wc wall cell ids."""
        e = e + self.edge(torch.cat((e, x[src], x[dst]), 1))
        w = w + self.wall(torch.cat((w, x[wc], s.expand(len(wc), -1)), 1))
        agg_e = torch.zeros_like(x).index_add_(0, dst, e)
        agg_w = torch.zeros_like(x).index_add_(0, wc, w)                       # structure -> wall cells
        x = x + self.node(torch.cat((x, agg_e, agg_w), 1))
        s = s + self.struct(torch.cat((s, w.mean(0, keepdim=True), w.sum(0, keepdim=True) / 64.0), 1))   # wall flow -> structure
        return x, e, w, s


class CoarseBlock(nn.Module):
    """Message passing on a coarse level (cluster graph): edge update from (edge, sender, receiver), node update; residual."""
    def __init__(self, h: int):
        super().__init__(); self.edge = mlp(3 * h, h, h); self.node = mlp(2 * h, h, h)

    def forward(self, x, e, src, dst):
        e = e + self.edge(torch.cat((e, x[src], x[dst]), 1))
        return x + self.node(torch.cat((x, torch.zeros_like(x).index_add_(0, dst, e)), 1)), e


class GlobalAttention(nn.Module):
    """Transformer layer over the coarsest graph level (~215 nodes): every cluster attends to every other one in one step
    (domain-wide interaction: both cylinders and the whole wake). Pre-LN; output projections zero-initialised (starts as identity)."""
    def __init__(self, h: int, heads: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, heads, batch_first=True); self.ff = mlp(h, h, h, norm=False)
        nn.init.zeros_(self.attn.out_proj.weight); nn.init.zeros_(self.attn.out_proj.bias); zero_last(self.ff)

    def forward(self, x, pe):
        q = self.ln1(x + pe)[None]
        x = x + self.attn(q, q, q, need_weights=False)[0][0]
        return x + self.ff(self.ln2(x))


class FSIGraphNet(nn.Module):
    NODE_IN, EDGE_IN, WALL_IN, STRUCT_IN = 28, 9, 8, 12
    FB_NODE, FB_EDGE, FB_WALL, FB_STRUCT = 3, 1, 5, 8

    def __init__(self, hidden: int = 128, stages: int = 3, layers: int = 2, levels: int = 3, coarse_layers: int = 2,
                 heads: int = 4, attention_layers: int = 1, memory_dim: int = 64, checkpoint: bool = True):
        super().__init__()
        if layers < 2: raise ValueError('layers must be >= 2 (fine blocks before and after the coarse path)')
        h = hidden
        self.stages, self.layers, self.levels, self.coarse_layers, self.memory_dim = stages, layers, levels, coarse_layers, memory_dim
        self.use_checkpoint = checkpoint
        self.pos_enc = mlp(2, h, h, norm=False)
        self.gattn = nn.ModuleList(nn.ModuleList(GlobalAttention(h, heads) for _ in range(attention_layers)) for _ in range(stages))
        # per-cell temporal memory (GRU); mem_in starts at zero output
        self.mem_in = mlp(memory_dim, h, h, norm=False); zero_last(self.mem_in); self.mem_cell = nn.GRUCell(h, memory_dim)
        self.enc_node, self.enc_edge = mlp(self.NODE_IN, h, h), mlp(self.EDGE_IN, h, h)
        self.enc_wall, self.enc_struct = mlp(self.WALL_IN, h, h), mlp(self.STRUCT_IN, h, h)
        # feedback encoders: no LayerNorm (they start at exactly zero output; LayerNorm of 0 has a ~1/sqrt(eps) gradient)
        self.fb_node, self.fb_edge = mlp(self.FB_NODE, h, h, norm=False), mlp(self.FB_EDGE, h, h, norm=False)
        self.fb_wall, self.fb_struct = mlp(self.FB_WALL, h, h, norm=False), mlp(self.FB_STRUCT, h, h, norm=False)
        self.blocks = nn.ModuleList(Block(h) for _ in range(stages * layers))
        self.enc_coarse = nn.ModuleList(mlp(3, h, h) for _ in range(levels))                       # coarse edge geometry
        self.coarse = nn.ModuleList(nn.ModuleList(CoarseBlock(h) for _ in range(levels * coarse_layers)) for _ in range(stages))
        self.up = nn.ModuleList(nn.ModuleList(mlp(2 * h, h, h) for _ in range(levels)) for _ in range(stages))
        self.dec_cell = mlp(h, h, 3, norm=False)            # [dU_x, dU_y, dp]
        self.dec_face = mlp(3 * h, h, 1, norm=False)        # interior face flux correction (antisymmetric)
        self.dec_wall = mlp(2 * h, h, 1, norm=False)        # rear-wall face flux correction
        self.dec_struct = mlp(h + 3, h, 3, norm=False)      # structure correction; sees the wall force of this stage's flow
        for d in (self.dec_cell, self.dec_face, self.dec_wall, self.dec_struct): zero_last(d)
        for f in (self.fb_node, self.fb_edge, self.fb_wall, self.fb_struct): zero_last(f)

    def encode(self, node, edge, wall, struct):
        return self.enc_node(node), self.enc_edge(edge), self.enc_wall(wall), self.enc_struct(struct)

    def feedback(self, x, e, w, s, fb):
        return x + self.fb_node(fb['node']), e + self.fb_edge(fb['edge']), w + self.fb_wall(fb['wall']), s + self.fb_struct(fb['struct'])

    def _run(self, f, *a):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(f, *a, use_reentrant=False)
        return f(*a)

    def encode_coarse(self, levels):
        """levels: [(assign, src, dst, counts, edge_geom, pos)] -> encoded coarse edge latents per level."""
        return [self.enc_coarse[l](lv[4]) for l, lv in enumerate(levels)]

    def process(self, stage: int, x, e, w, s, src, dst, wc, levels, coarse_e):
        """One coupling stage: fine blocks -> pool through the coarse levels (message passing on each, global attention on
        the coarsest) -> unpool level by level with skip connections -> remaining fine blocks. Fixed depth."""
        fine = self.blocks[stage * self.layers:(stage + 1) * self.layers]; half = len(fine) // 2
        for blk in fine[:half]: x, e, w, s = self._run(blk, x, e, w, s, src, dst, wc)
        cl = self.coarse_layers; stack = [x]; cur = x
        for l, (assign, cs, cd, cnt, _, pos) in enumerate(levels):
            cur = torch.zeros(len(cnt), cur.shape[1], dtype=cur.dtype, device=cur.device).index_add_(0, assign, cur) / cnt[:, None]
            ce = coarse_e[l]
            for blk in self.coarse[stage][l * cl:(l + 1) * cl]: cur, ce = self._run(blk, cur, ce, cs, cd)
            if l == len(levels) - 1:
                pe = self.pos_enc(pos)
                for layer in self.gattn[stage]: cur = self._run(layer, cur, pe)
            stack.append(cur)
        for l in range(len(levels) - 1, -1, -1):
            assign = levels[l][0]; finer = stack[l]
            stack[l] = finer + self._run(self.up[stage][l], torch.cat((finer, stack[l + 1][assign]), 1))
        x = stack[0]
        for blk in fine[half:]: x, e, w, s = self._run(blk, x, e, w, s, src, dst, wc)
        return x, e, w, s

    def mem_inject(self, x, mem):
        return x + self.mem_in(mem)

    def mem_update(self, x, mem):
        return self.mem_cell(x, mem)

    def decode_flow(self, x, e, w, s, src, dst, n_int, wc):
        cell = self.dec_cell(x)
        fwd = self.dec_face(torch.cat((x[src[:n_int]], x[dst[:n_int]], e[:n_int]), 1))
        bwd = self.dec_face(torch.cat((x[src[n_int:]], x[dst[n_int:]], e[n_int:]), 1))
        wall = self.dec_wall(torch.cat((w, x[wc]), 1))[:, 0]
        return cell, (fwd - bwd)[:, 0], wall

    def decode_struct(self, s, force_feat):
        return self.dec_struct(torch.cat((s, force_feat[None, :]), 1))[0]
