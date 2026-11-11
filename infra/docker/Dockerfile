# synthrig runtime image: Python 3.11 + Blender 4.2 LTS (pip `bpy` wheel, option A of
# docs/TASK_infra.md step 0.1) + the repo's locked dependencies. One image is used for
# tests, GPU checks and rendering. Build with:  python infra/cloud.py image build
#
# GPU: run with `--gpus all` (or `--runtime nvidia`). Cycles uses the host driver's
# CUDA/OptiX libraries, which the NVIDIA container toolkit mounts in at run time, so the
# small CUDA *base* image is enough. Without a GPU, Cycles falls back to CPU.
ARG CUDA_BASE=nvidia/cuda:12.4.1-base-ubuntu22.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.5

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------- build stage
FROM ${CUDA_BASE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1
WORKDIR /src
COPY pyproject.toml uv.lock ./
# Locked deps + Blender (render extra) + GUI (gui extra) + pytest (dev group). The project itself is
# not installed; /code is put on PYTHONPATH in the runtime stage.
RUN uv sync --frozen --extra render --extra gui --no-install-project

# ---------------------------------------------------------------- runtime stage
FROM ${CUDA_BASE}
# Headless runtime libraries bpy links against (no build toolchain in this layer).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 \
      libsm6 libice6 libgl1 libegl1 libgomp1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/venv /opt/venv

ARG GIT_SHA=unknown
ARG GIT_DIRTY=unknown
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/code \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    SYNTHRIG_DATA_ROOT=/data \
    SYNTHRIG_GIT_SHA=${GIT_SHA} \
    SYNTHRIG_GIT_DIRTY=${GIT_DIRTY}
WORKDIR /code
COPY . /code
# Workers run as the host user (--user), so the code must be world-readable.
RUN chmod -R a+rX /code && chmod a+x /code/infra/docker/entrypoint.sh
ENTRYPOINT ["/code/infra/docker/entrypoint.sh"]
CMD ["--help"]
