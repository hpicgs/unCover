FROM continuumio/miniconda3
LABEL authors="lucasliebe"

RUN apt-get update -y && apt-get install default-jre make wget curl unzip build-essential libssl-dev -y
COPY . /app/unCover
WORKDIR /app/unCover

# conda env
RUN conda update conda --yes
RUN conda env create -f environment.yml
RUN conda run -n unCover pip install -r tem/script/requirements.txt
RUN conda run -n unCover pip install cmake

# TEM build & setup
RUN rm tem/build/ -fr && mkdir tem/build/ && cd tem/build/ && conda run -n unCover cmake .. && make
RUN conda run -n unCover python -m nltk.downloader punkt
RUN conda run -n unCover python -m nltk.downloader stopwords
RUN conda run -n unCover python -m spacy download en_core_web_md
RUN conda run -n unCover python -m spacy download de_core_news_md

# final container setup
RUN cp /opt/conda/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.28
RUN conda run -n unCover ./deployment/corenlp --no-run
