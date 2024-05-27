# Builder image
FROM lucasliebe/uncover-training as builder

# prepare term-distance server
RUN conda create -n tem-term-distance python=3.8.19
RUN conda run -n tem-term-distance pip install -r tem/term-distance/requirements.txt
RUN --mount=type=cache,target=/root/gensim-data conda run -n tem-term-distance python tem/term-distance/model.py

# download pre-trained models
RUN conda run -n unCover ./deployment/prepare_models

# Final image
FROM continuumio/miniconda3
LABEL authors="lucasliebe"

COPY --from=builder /app /app
COPY --from=builder /opt/conda /opt/conda

WORKDIR /app/unCover
CMD conda run -n tem-term-distance ./tem/term-distance/run.sh && conda run -n unCover streamlit run main.py
