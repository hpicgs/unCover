# Builder image
FROM ubuntu:latest as builder
LABEL authors="lucasliebe"


RUN apt-get update -y && apt-get install default-jre make wget curl unzip build-essential -y
COPY . /app/unCover

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p $CONDA_DIR
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app/unCover
RUN conda env create -f environment.yml
RUN cp -n .env.example .env
RUN make -C tem/topic-evolution-model/
RUN conda run -n unCover ./corenlp --no-run && conda run -n unCover ./prepare_models

# Final image
FROM scratch
LABEL authors="lucasliebe"

#RUN apt-get update -y && apt-get install default-jre
COPY --from=builder /app/unCover /app/unCover
ENV CONDA_DIR /opt/conda
COPY --from=build $CONDA_DIR $CONDA_DIR
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app/unCover
CMD conda run -n unCover streamlit run main.py