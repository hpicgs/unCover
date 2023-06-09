# XAI Project

Download [Stanford's CoreNLP
4.5.1](https://stanfordnlp.github.io/CoreNLP/index.html), unzip it and place it
into the `models` directory.
Download NLTK's `punkt`: `nltk.download("punkt")` and `nltk.download("stopwords")`

To be able to start processing a text please first run run_CoreNLP_Server.sh and
then run the desired scripts in another terminal.  The server can be stopped
through ^C in the original terminal.

## Scraping

Scraping can be done either in query mode or dataset mode - query mode uses a
search query provided by you to gather articles on google news, while dataset
mode looks up the personal pages of specific authors on their publication and
collects hundreds of articles for a single person.

### Query mode

Query mode is pretty straightforward:

```shell
python3 scrape_authorized_articles.py --query <your_query>
```

Some queries fetch no results on news.google.com, the scraper will break after
10 seconds in that case.

### Dataset mode

Dataset mode is quicker and has higher output but takes a bit more work on your
part. For one, every news publication has their own way of listing articles by
author, if at all, therefore we need to write helper functions for every single
publication. At the moment, dataset mode only works for our favourite newspaper,
TheGuardian.

Here is a practical example for dataset mode:

```shell
python3 scrape_authorized_articles.py \
    --dataset \
    --publication theguardian.com \
    --author https://www.theguardian.com/profile/martin-chulov \
    --narticles 300
```

The publication argument will activate dataset creation for TheGuardian, you
should provide the URL of theguardian.com here. The author argument is the most
important one - you need to provide the (full) URL of your specific author's
personal page on theguardian.com, where all their articles are listed.
