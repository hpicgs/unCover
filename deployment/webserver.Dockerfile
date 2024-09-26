# Builder image
FROM lucasliebe/uncover-training AS builder
COPY .env /app/unCover/.env

# prepare term-distance server
RUN conda create -n tem-term-distance python=3.8.19
RUN conda run -n tem-term-distance pip install -r tem/term-distance/requirements.txt
WORKDIR /app/unCover/tem/term-distance
RUN conda run -n tem-term-distance python model.py

# download pre-trained models
WORKDIR /app/unCover
RUN conda run -n unCover ./deployment/prepare_models

# Final image
FROM continuumio/miniconda3
LABEL authors="lucasliebe"

COPY --from=builder /app /app
COPY --from=builder /root/gensim-data /root/gensim-data
COPY --from=builder /opt/conda/envs /opt/conda/envs

WORKDIR /app/unCover
CMD ["sh", "-c", "cd tem/term-distance && conda run -n tem-term-distance ./run.sh & sleep 1; cd /app/unCover && sleep 10 && conda run --no-capture-output -n unCover streamlit run main.py"]
