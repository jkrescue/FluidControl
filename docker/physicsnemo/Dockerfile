FROM nvcr.io/nvidia/physicsnemo/physicsnemo@sha256:6fafbf1ca17ac726182d67946e2c0f34061ffaf22b8754df2c3fabe7bd52ceac

# Preserve the GB10-ready CUDA/PyTorch stack and upgrade only PhysicsNeMo.
RUN python -m pip install --no-cache-dir --no-deps nvidia-physicsnemo==2.2.2

WORKDIR /workspace
