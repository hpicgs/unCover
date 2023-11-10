FROM ubuntu:latest
LABEL authors="lucasliebe"

WORKDIR /app

RUN apt-get update -y && apt-get install default-jre make git wget curl unzip build-essential -y
RUN git clone --recurse-submodules https://github.com/hpicgs/unCover

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app/unCover
RUN conda init && conda env create -f environment.yml
RUN cp .env.example .env
RUN make -C tem/topic-evolution-model/
RUN conda run -n unCover ./corenlp --no-run && conda run -n unCover ./prepare_models

CMD conda run -n unCover streamlit run main.py