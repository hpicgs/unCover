# Builder image
FROM continuumio/miniconda3 as builder

RUN apt-get update -y && apt-get install default-jre make wget curl unzip build-essential -y
COPY . /app/unCover

WORKDIR /app/unCover
RUN conda update conda --yes
RUN conda env create -f environment.yml
RUN cp -n .env.example .env
RUN make -C tem/topic-evolution-model/
RUN conda run -n unCover ./corenlp --no-run && conda run -n unCover ./prepare_models

# Final image
FROM continuumio/miniconda3
LABEL authors="lucasliebe"

COPY --from=builder /app /app
COPY --from=builder /opt/conda /opt/conda

WORKDIR /app/unCover
CMD conda run -n unCover streamlit run main.py