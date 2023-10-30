# unCover

Detailed information about unCover can be found in the following publication:

> Liebe L, Baum J, Schutze T, Cech T, Scheibel W, and Dollner J (2023). UNCOVER:
> Identifying AI Generated News Articles by Linguistic Analysis and
> Visualization

## Setup

<!--
TODO: do we want to include instructions to deploy on streamlit?
-->

To setup this project to run on your own machine, run the following command

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

<!--
TODO: tell people to set up .env
-->

## Usage

<!--
TODO: introductory paragraph for different usage modes (running the server,
scraping, etc
-->

<!--
TODO: include instructions for running the server normally
-->

### Scraping

<!--
TODO: check if this is correct & up to date
-->

Scraping can be done either in query mode or dataset mode - query mode uses a
search query provided by you to gather articles on google news, while dataset
mode looks up the personal pages of specific authors on their publication and
collects hundreds of articles for a single person.

#### Query mode

Query mode is pretty straightforward:

```shell
python3 generate_authorized_articles_dataset.py --query <your_query>
```

Some queries fetch no results on news.google.com, the scraper will break after
10 seconds in that case.

#### Dataset mode

Dataset mode is quicker and has higher output but takes a bit more work on your
part. For one, every news publication has their own way of listing articles by
author, if at all, therefore we need to write helper functions for every single
publication. At the moment, dataset mode only works for our favourite newspaper,
TheGuardian.

Here is a practical example for dataset mode:

```shell
python3 generate_authorized_articles_dataset.py \
    --dataset \
    --publication theguardian.com \
    --author https://www.theguardian.com/profile/martin-chulov \
    --narticles 300
```

The publication argument will activate dataset creation for TheGuardian, you
should provide the URL of theguardian.com here. The author argument is the most
important one - you need to provide the (full) URL of your specific author's
personal page on theguardian.com, where all their articles are listed.
