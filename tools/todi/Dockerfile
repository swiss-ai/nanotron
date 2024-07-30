FROM nvcr.io/nvidia/pytorch:24.05-py3

# Setup
RUN apt-get update && apt-get install python3-pip python3-venv -y
RUN pip install --upgrade pip setuptools==69.5.1

RUN pip install flash-attn==2.5.8 --no-build-isolation

COPY nanotron/ /workspace/nanotron
WORKDIR /workspace/nanotron
RUN pip install -e '.[nanosets]'

# Instructions:
# 1. Build image: podman build -f /users/asolergi/SFT/nanotron/tools/todi/Dockerfile -t nanotron_sft /users/asolergi/SFT/ #### NOTE In /users/asolergi/SFT/ we have nanotron/ (/users/asolergi/SFT/nanotron)
# 2. Export image: enroot import -o /store/swissai/a06/.sft_toni/nanotron_sft.sqsh podman://localhost/nanotron_sft:latest
