# protein-transformer
this repository contains code to build a transformer model with sequences of amino acids which proteins in human body are made by.

the goal of this program is to build transformer model and train it with sequences of amino acids in human body and calculate the accuracy of prediciting the next token in a given sequence
the dataset, sequences of amino acids, has been collected manually from https://www.uniprot.org website in fasta format and appended to pandas.DataFrame object with its corresponding gene and protein name , for now it only contains 4 major protein in human body. 
limitations: it has few sequences of amino acids

# the program
1. run ./env.sh # creates virtual environemt and install required packages for ubuntu 20.04
2. activate the virtual environment: source [the directory you cloned the repo]/venv/protein/bin/activate
3. run ./main.py
