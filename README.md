# unCover

Detailed information about unCover can be found in the following publication:

> Liebe L., Baum J., Schütze T., Cech T., Scheibel W. and Döllner J. (2023).  
> UNCOVER: Identifying AI Generated News Articles by Linguistic Analysis and Visualization.  
> In Proceedings of the 15th International Joint Conference on Knowledge Discovery, 
> Knowledge Engineering and Knowledge Management - Volume 1: KDIR; ISBN
> 978-989-758-671-2, SciTePress, pages 39-50. DOI: 10.5220/0012163300003598   

![Teaser](https://drive.google.com/uc?export=download&id=1DU9HwazIUGxoFdI5cJ-liW3Q_-a-QV6G)

An interactive example deployment of unCover can be found at 
[uncover.lucasliebe.de](https://uncover.lucasliebe.de).
Our datasets and pre-trained models can be found in 
[Google Drive](https://drive.google.com/drive/folders/1fMZgGC2Bnp5K-ZoANXB_S0AI02akye_c?usp=drive_link).
Please note that for copyright reasons we removed the plain text of the scraped 
news articles and only left the metadata and the generated texts in the dataset files.

## Prerequisites

Before you can use the installation script as described below, please make sure
you have the following packages installed and working on your machine:
- Anaconda or Miniconda
- Java (Runtime Environment is sufficient)
- Make and g++

## Setup

To set up this project to run on your own machine, run the following command

```sh
sh <(curl -s https://raw.githubusercontent.com/hpicgs/unCover/main/install.sh)
```

This will download and install all requirements to an
[Anaconda](https://www.anaconda.com) environment.

Then, whenever you want to run any of the following scripts, make sure you first
activate the environment with

```sh
conda activate unCover
```

To take full advantage of all capabilities in this repository you should fill out 
all the information in `.env.example` and save it as `.env`. OpenAI-credentials are
only required for generation, however, fine-tuning the confidence thresholds to the 
used models will greatly benefit performance and is required to achieve good results.

Alternatively if you are only interested as running the web interface you can use
the provided docker container for a quick deployment. It can be built yourself using 
the Dockerfile or pulled from Docker Hub: `docker pull lucasliebe/uncover:latest`.

## Usage

There are multiple ways to use unCover, depending on your use case.
First, you can run the server and use the web interface to explore the system.
Second, you can generate your own news dataset following the proposed methods.
Third, you can use the datasets to re-train and/ or test the unCover models.

### Running unCover

To run and try out unCover in its base configuration follow the setup guide and run

```
streamlit run main.py
```

This will start the process and ensure all sub-systems are installed and running, 
then you can access the web interface locally at `http://localhost:8501/`.

### Scraping news articles

Scraping can be done either in query mode or dataset mode - query mode uses a
search query provided by you to gather articles on Google News, while dataset
mode looks up the personal pages of specific authors on their publication and
collects hundreds of articles for a single person.

#### Query mode

Query mode is very simple and will fetch news from Google News depending on the query:

```shell
python3 generate_authorized_articles_dataset.py --query <your_query>
```

#### Dataset mode

Dataset mode is quicker and has higher output but takes a bit more work to set up. 
For one, every news publication has their own way of listing articles by
author, therefore a helper functions is required for every single
publication. At the moment, dataset mode only works for TheGuardian.

Here is a practical example for dataset mode:

```shell
python3 generate_authorized_articles_dataset.py \
    --dataset \
    --publication theguardian.com \
    --author https://www.theguardian.com/profile/<author name> \
    --narticles 300
```

The publication argument will activate dataset creation for TheGuardian, you
should provide the URL of theguardian.com here. The author argument is the most
important one - you need to provide the (full) URL of your specific author's
personal page on theguardian.com, where all their articles are listed.

### Generating news articles

To generate news articles, we also scrape news from Google News, but this time
we use the query as context for a generative language model. The model will
then generate a new article based on this context. The following could is an example command:

```shell
python3 generate_news_articles.py --queries <your queries> --grover base --gpt3
```

### Training the models

New models can be trained using `train_authors.py` for stylometry models and 
`train_tem_metrics.py` for the TEM based prediction model. The commandline options
for these scripts are documented in the scripts themselves. If your dataset is stored
in a different location, please modify `definitions.py` accordingly.

### Recreating the test performance

Including both former methods, `generate_test_dataset.py` has been used to create
the test dataset including both human authors and AI models as text sources.

After creating this test dataset or using the provided dataset, 
`test_performance.py` can be used to create test results and a performance summary.

## Citation

If you use unCover in your research, please cite our paper:

```bibtex
@inproceedings{kdir23uncover,
    author={Lucas Liebe and Jannis Baum and Tilman Schütze and Tim Cech and Willy Scheibel and Jürgen Döllner},
    title={UNCOVER: Identifying AI Generated News Articles by Linguistic Analysis and Visualization},
    booktitle={Proceedings of the 15th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management - Volume 1: KDIR},
    year={2023},
    pages={39-50},
    publisher={SciTePress},
    organization={INSTICC},
    doi={10.5220/0012163300003598},
    isbn={978-989-758-671-2},
}
```

