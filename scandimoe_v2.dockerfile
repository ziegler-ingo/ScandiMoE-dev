#FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS base

WORKDIR /workspace

RUN apt update && \
    apt install -y python3-pip python3-packaging \
    git ninja-build && \
    pip3 install -U pip

# 8.9 for NVIDIA GeForce RTX 4090, 9.0 for H100
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0" 
RUN pip3 install torch==2.4.0
RUN pip3 install numpy

RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124 
RUN pip3 install transformers
RUN pip3 install deepspeed
RUN pip3 install flash-attn --no-build-isolation

COPY entrypoint.sh .
COPY src/ src/

RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]