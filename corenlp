#!/bin/sh

# make sure we have java
if ! which java 1>/dev/null 2>/dev/null; then
    echo "Please ensure java is installed and on your PATH."
    exit 1
fi

# find & go to appropriate model directory (in conda env or repo)
if [ -n "$CONDA_PREFIX" ]; then
    MODEL_DIR="$CONDA_PREFIX/models"
else
    SCRIPT="`readlink -f $0`"
    REPO_DIR="`dirname $SCRIPT`"
    MODEL_DIR="$REPO_DIR/.models"
fi
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# download stanford parser if not there
PARSER_DIR=`ls | grep 'stanford-corenlp-4.5' | tail -1`
if [ -z "$PARSER_DIR" ]; then
    curl -o download.zip https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-4.5.5.zip
    unzip download.zip
    rm download.zip
    PARSER_DIR="stanford-corenlp-4.5.5"
fi

# run parser
cd "$PARSER_DIR"
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 150000
