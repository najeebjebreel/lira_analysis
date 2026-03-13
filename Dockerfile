# syntax=docker/dockerfile:1
# LiRA Analysis — PoPETs 2026 artifact
# Requires: NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
# Build:    docker build -t lira-analysis:main .
# Run:      docker run --gpus all --rm -it \
#               -v $(pwd)/data:/workspace/data \
#               -v $(pwd)/experiments:/workspace/experiments \
#               -v $(pwd)/analysis_results:/workspace/analysis_results \
#               lira-analysis:main bash

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget ca-certificates git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda ─────────────────────────────────────────────────────────────────
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-Linux-x86_64.sh \
        -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# ── Conda environment (pinned via environment.yml, minus the editable install) ─
WORKDIR /workspace
COPY environment.yml requirements-lock.txt ./
# Strip the "-e ." editable-install line so conda env create doesn't need
# the source tree; we install the package properly after COPY . .
RUN sed '/^\s*-\s*-e \./d' environment.yml > environment_docker.yml && \
    conda env create -f environment_docker.yml && \
    conda clean -afy && \
    rm environment_docker.yml

# ── Copy source (data/, experiments/, analysis_results/ are mounted at runtime) ──
COPY . .

# ── Editable install now that source is present ───────────────────────────────
RUN conda run -n lira-repro pip install -e . --no-deps

# ── Default shell inside the lira-repro environment ──────────────────────────
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "lira-repro"]
CMD ["bash"]
