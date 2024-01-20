# Latent Semantic Indexing project
Piero Pettenà - January 2024

This is a Python project for the course of Information Retrieval in DSSC Master Degree in University of Trieste.
A simple, educational-purposed information retrieval system is set up and tested on TIME magazine dataset.

This file illustrates how the directory is organized and how to execute the code.

## Requirements
The code has been written using:

- python 3.12.1
- pandas 2.1.4
- numpy 1.26.3
- scipy 1.11.4
- nltk 3.8.1
- scikit-learn 1.3.2

## Directory organization
The executable file is, of course, lsi.py.
The dataset can be found inside "data/time" folder, while saved document/term and TF-IDF matrices can be found inside "matrices" folder.
Matrix files whose names begin with "my" are calculated using my slow and not fully tested algorithm, while the ones beginning with "true" are based on the reliable scikit-learn library.

## Executing the code
To execute code, some options are available:

1) Normal execution:

    $ python lsi.py 

Takes a long time to execute (7-8 minutes on my computer) because the document/term and the TF-IDF matrices are being calculated by my algorithm. Once this is done, the user is asked to insert a query and this will be executed, reporting top 5 results. For the execution of the query, however, sklearn's TF-IDF matrix is used. It makes no practical sense to use this option apart from the educational purpose of implementing class notions. 

2) Fast execution:
    
    $ python lsi.py -s

Takes a few seconds to set up because uses scikit-learn library to calculate matrices. Then proceeds as option 1.

3) Verbose execution:

    $ python lsi.py -v

Prints additional information during the execution. It is recommended to add this option when using option 1.

4) Matrix comparison operation

    $ python lsi.py -c

Compares my matrices with the ones from scikit-learn. 

5) Any combination of the above
6) Help on command line arguments
   
   $ python lsi.py -h
