#!/bin/sh

cd models/stanford-corenlp-4.5.1 || exit
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
