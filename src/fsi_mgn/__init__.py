"""fsi_mgn: FSIGraphNet, a MeshGraphNet-based physics-constrained joint predictor for fluid-structure interaction.

One forward pass per time step outputs the flow (cell U, p, face flux) and the structure (y, v, a). No equation is solved:
`physics` only evaluates explicit formulas (wall-force integral, per-cell flux balance, wall-flux mismatch, structural and
kinematic residuals) on predicted quantities. These values are fed back into the network between its fixed coupling
stages and used as training-loss terms. The package imports no solver (tests/no_solver.py).
"""
