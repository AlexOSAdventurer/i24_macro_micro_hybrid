# Lightweight, multi-architecture (linux/amd64 + linux/arm64) image for the
# "Quick-Load" pipeline: macroscopic LWR + SUMO micro bubble, no CARLA / Unreal.
#
# Unlike container.Dockerfile this does not build on carla-base (x86_64-only), so it
# runs natively on Apple Silicon instead of under amd64 emulation. Torch is the CPU
# build; there is no GPU inside Docker on macOS anyway.
#
# Build (both architectures, push to a registry):
#   ./build_quick_container.sh --push
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# ffmpeg: needed by ffmpeg-python. libgomp1: OpenMP runtime for torch/scipy wheels.
# libspatialindex-dev: rtree's native backend. libatomic1: the eclipse-sumo wheel's
# binaries link it but do not bundle it.
# procps/htop/less/nano: basic interactive tools (top, ps, ...) the slim base lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 libspatialindex-dev libatomic1 \
        procps htop less nano \
    && rm -rf /var/lib/apt/lists/*

# Reuse the project's requirements, but swap the CUDA torch wheels (x86_64-only and
# several GB) for the CPU builds, which exist for both amd64 and arm64.
COPY i24motion_macro_micro/requirements.txt /tmp/requirements.txt
RUN sed -e 's#^--extra-index-url .*#--extra-index-url https://download.pytorch.org/whl/cpu#' \
        -e 's#+cu[0-9]*#+cpu#' \
        /tmp/requirements.txt > /tmp/requirements-quick.txt \
    && pip install -r /tmp/requirements-quick.txt \
    && pip install eclipse-sumo==1.27.1 traci==1.27.1 sumolib==1.27.1 scipy \
    && rm /tmp/requirements*.txt

# eclipse-sumo installs its binaries inside site-packages; expose them and SUMO_HOME
# so sumolib.checkBinary("sumo") finds them.
RUN SUMO_HOME="$(python -c 'import sumo, os; print(os.path.dirname(sumo.__file__))')" \
    && ln -s "$SUMO_HOME" /opt/sumo \
    && if ldd /opt/sumo/bin/* 2>/dev/null | grep "not found"; then \
         echo "SUMO binaries have unresolved shared libraries (above)" >&2; exit 1; \
       fi \
    && /opt/sumo/bin/sumo --version > /dev/null
ENV SUMO_HOME=/opt/sumo

# The launch script runs as the host UID, which has no passwd entry or home here.
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspaces/i24motion_macro_micro
EXPOSE 8050
CMD ["bash"]
