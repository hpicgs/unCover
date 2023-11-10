# unCover

Detailed information about unCover can be found in the following publication:

> Liebe L, Baum J, Schutze T, Cech T, Scheibel W, and Dollner J (2023). UNCOVER:
> Identifying AI Generated News Articles by Linguistic Analysis and
> Visualization

![Teaser](https://drive.google.com/uc?export=download&id=1i49F16U7TiHCS8-17lBv8ofPsnvd-RE0)

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

