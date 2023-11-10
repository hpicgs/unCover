#!/bin/sh

print_bold() {
    printf "\\033[1m$1\\033[0m\n"
}


if ! which conda 1>/dev/null 2>/dev/null; then
    print_bold "Please ensure Anaconda is installed and on your PATH."
    exit 1
fi

if ! which java 1>/dev/null 2>/dev/null; then
    print_bold "Please ensure Java is installed and on your PATH."
    exit 1
fi

if ! which make 1>/dev/null 2>/dev/null; then
    print_bold "Please ensure Make is installed and on your PATH."
    exit 1
fi

print_bold "Cloning repository"
git clone --recurse-submodules https://github.com/hpicgs/unCover.git
cd unCover

print_bold "Creating Anaconda environment"
conda init
conda env create -f environment.yml

print_bold "Compiling TEM"
make -C tem/topic-evolution-model/ # make: g++: No such file or directory

print_bold "Installing CoreNLP"
conda run -n unCover ./corenlp --no-run

print_bold "Downloading Models"
conda run -n unCover ./prepare_models

print_bold "Done!"
