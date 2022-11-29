#!/bin/sh

cd models/stanford-corenlp-4.5.1
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 &
cd ../..
python train_authors.py
wget "localhost:9000/shutdown?key=`cat /tmp/corenlp.shutdown`" -O -