FROM fluid-control-physicsnemo:2.2.2

# PhysicsNeMo Curator source commit: 86533e581b3550326d89e97cb4d4126e7061b416
# Wheel SHA-256: 903547bc127827c5b7b3ba0e4bef8a42ba8a78b316613d9919de00463311d593
COPY physicsnemo_curator-0.1.0-cp312-abi3-linux_aarch64.whl /tmp/
RUN python -m pip install --no-deps /tmp/physicsnemo_curator-0.1.0-cp312-abi3-linux_aarch64.whl \
    && python -c 'from physicsnemo_curator.core.base import Source, Filter, Sink; from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource; from physicsnemo_curator.run import run_pipeline; from physicsnemo.mesh.spatial import BVH; print("CURATOR_IMPORT_OK")' \
    && python -m pip freeze | grep -E '^(nvidia-physicsnemo|physicsnemo-curator|vtk|pyvista)=='
