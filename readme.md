## Human-Level Control through Deep Reinforcement Learning (paper id: 63)

### Team Members:
* Kumar Anant Raj : 2016B4A70520P
* Akash Kabra     : 2016B3A70562P

### Directory Structure

|__config.py
|__Corpus_Preprocessing.py
|__test_queries.py
|__Spell_Checked.py
|__wiki_00

Please note that the order of running the python scripts should be maintained as specified as all of them are interdependent scripts. 

### System Requirements 

The following library functions are used. Please install these libraries if they are not available in your system 

bs4, nltk, math, operator, stringdist, string, json, pickle, io, os, sys, types, dill

### Running Guidelines

***********************************************************
PART 1:
***********************************************************

1. run config.py 
PRE-CONDITION:
File "wiki_00" or any other file should be present in the directory

Please note that for Part 1, we don't wish to do Stem and Lemmatize, so the
self.STEMMING = False
self.LEMMATIZE = False
and make the output file names as required (or maybe, keep it as default)

POST-CONDITION:
A file "config.pkl" should be made in the directory

Purpose:
This file would initialize all the global variables like output file names etc. which would be used by other python files. 


2. run Corpus_Preprocessing.py
PRECONDITION: 
A file "config.pkl" should be present in the directory

POST-CONDITION:
4 json files by the name specified in "config.py" should be made in the directory. 

Purpose: 
This file would preprocess the corpus, and gives the inverse mapping required for test_queries.py 
This should take 3-5 minutes to run. 


3. run test_queries.py
PRECONDITION:
If you want to update the query, update the "query" variable in main function. 
config.pkl and 4 other json files by the name specified in "config.py" should be present in the directory. 

POSTCONDITION:
You must have received top 10(or less than top 10 if the query is relevant to less than 10 documents) documents for the query on console. 

Purpose: 
This file tests queries by using the preprocessing made. 


***********************************************************
PART 2:
***********************************************************

We have introduced Spell Check using weighted levenstein and soundex distance, stemming and lemmatization on the corpus to increase recall of the IR system. 
To run for Stemming + Spell Check implementation: 

1. Open Config.py
Modify it as follows:
        self.STEMMING = True
        self.LEMMATIZATION = False
        self.TF_LIST = "tf_list_stem.json" // Change the file name
        self.IDF_DICT  = "idf_dict_stem.json" // Change the file name
        self.INVERSE_MAPPING = "inverse_mapping_stem.json" // Change the file name
        self.DOC_TITLE_LIST = "doc_title_list_stem.json"  // Change the file name
        self.FILENAME = "wiki_00"

run config.py:
PRE-CONDITION:
File "wiki_00" or any other file should be present in the directory

POST-CONDITION:
A file "config.pkl" should be made in the directory

2. run Corpus_Preprocessing.py
PRECONDITION: 
A file "config.pkl" should be present in the directory

POST-CONDITION:
4 json files by the name specified in "config.py" should be made in the directory. 


3. run test_queries.py
PRECONDITION:
If you want to update the query, update the "query" variable in main function. 
config.pkl and 4 other json files by the name specified in "config.py" should be present in the directory. 

POSTCONDITION:
You must have received top 10(or less than top 10 if the query is relevant to less than 10 documents) documents for the query on console. 



To run for Lemmatization + Spell Check implementation: 

Please note that this may take more time than the previous two implementations, as lemmatization involves dictionary lookup. 

Modify it as follows:
        self.STEMMING = False
        self.LEMMATIZATION = True
        self.TF_LIST = "tf_list_lem.json" // Change the file name
        self.IDF_DICT  = "idf_dict_lem.json" // Change the file name
        self.INVERSE_MAPPING = "inverse_mapping_lem.json" // Change the file name
        self.DOC_TITLE_LIST = "doc_title_list_lem.json"  // Change the file name
        self.FILENAME = "wiki_00"

Other steps are same as the last implementation. 


