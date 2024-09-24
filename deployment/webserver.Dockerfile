# Builder image
FROM lucasliebe/uncover-training as builder

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
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/gensim-data /root/gensim-data

WORKDIR /app/unCover
CMD cd tem/term-distance && conda run -n tem-term-distance ./run.sh & sleep 1; cd /app/unCover && sleep 10 && conda run -n unCover streamlit run main.py
