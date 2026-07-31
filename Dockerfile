FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
ARG PICAI_PREP_REV=9544f73c8c4f986d9a2b7e2fe7f7b3ba717c3d5b
ARG PICAI_EVAL_REV=81d6067130cb264312ebcca220dd0784f074ae0e
ARG MEDCAM_REV=fddd001a6da20229d33a0cb7c5d594fa432763d9

ENV det_data=/opt/data \
    det_models=/opt/models \
    det_num_threads=6 \
    det_verbose=1 \
    OMP_NUM_THREADS=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN rm -f /etc/apt/sources.list.d/cuda*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && conda create -y -n gcalf python=3.8 pip \
    && conda clean -afy

ENV PATH=/opt/conda/envs/gcalf/bin:${PATH}

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113

RUN git clone https://github.com/DIAGNijmegen/picai_prep.git /opt/tools/picai_prep \
    && git -C /opt/tools/picai_prep checkout --detach ${PICAI_PREP_REV} \
    && git clone https://github.com/DIAGNijmegen/picai_eval.git /opt/tools/picai_eval \
    && git -C /opt/tools/picai_eval checkout --detach ${PICAI_EVAL_REV} \
    && git -C /opt/tools/picai_eval submodule update --init --recursive \
    && git clone https://github.com/MECLabTUDA/M3d-Cam.git /opt/tools/M3d-Cam \
    && git -C /opt/tools/M3d-Cam checkout --detach ${MEDCAM_REV}

WORKDIR /workspace/GCALF-Net
COPY . .

RUN mkdir -p ${det_data} ${det_models} \
    && python -m pip install --no-cache-dir -r requirements.txt pytest

RUN FORCE_CUDA=1 python -m pip install --no-build-isolation -v -e .

RUN python -m pip install --no-cache-dir \
    -e /opt/tools/picai_prep \
    -e /opt/tools/picai_eval \
    -e /opt/tools/M3d-Cam

ENV LD_LIBRARY_PATH=/opt/conda/envs/gcalf/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH}

RUN python -c "import medcam, nndet, nndet._C, picai_eval, picai_prep"

CMD ["bash"]
