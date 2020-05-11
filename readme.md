## Information Retrieval System using tf-idf query vectorization

### Directory Structure

|__config.py<br />
|__Corpus_Preprocessing.py<br />
|__test_queries.py<br />
|__Spell_Checked.py<br />
|__wiki_00<br /><br />

- Please note that the order of running the python scripts should be maintained as specified as all of them are interdependent scripts. 

### System Requirements 

The following library functions are used. Please install these libraries if they are not available in your system 

bs4, nltk, math, operator, stringdist, string, json, pickle, io, os, sys, types, dill

### Running Guidelines

***********************************************************
PART 1:
***********************************************************

**1. run config.py** <br />
**PRE-CONDITION:**<br />
File "wiki_00" or any other file should be present in the directory<br /><br />

Please note that for Part 1, we don't wish to do Stem and Lemmatize, so the
self.STEMMING = False
self.LEMMATIZE = False
and make the output file names as required (or maybe, keep it as default)<br />

**POST-CONDITION:**<br />
A file "config.pkl" should be made in the directory<br />

Purpose:<br />
This file would initialize all the global variables like output file names etc. which would be used by other python files<br />


**2. run Corpus_Preprocessing.py<br />
PRECONDITION: **<br />
A file "config.pkl" should be present in the directory<br />

**POST-CONDITION:**<br />
4 json files by the name specified in "config.py" should be made in the directory. <br />

Purpose: <br />
This file would preprocess the corpus, and gives the inverse mapping required for test_queries.py <br />
This should take 3-5 minutes to run. <br /><br />


**3. run test_queries.py**<br />
**PRECONDITION:**<br />
If you want to update the query, update the "query" variable in main function. <br />
config.pkl and 4 other json files by the name specified in "config.py" should be present in the directory.<br /><br /> 

**POSTCONDITION:**<br />
You must have received top 10(or less than top 10 if the query is relevant to less than 10 documents) documents for the query on console. <br />

Purpose: <br />
This file tests queries by using the preprocessing made. <br /><br />


***********************************************************
PART 2:
***********************************************************

We have introduced Spell Check using weighted levenstein and soundex distance, stemming and lemmatization on the corpus to increase recall of the IR system. <br />
To run for Stemming + Spell Check implementation: <br /><br />

**1. Open Config.py**<br />
Modify it as follows:<br />
        self.STEMMING = True<br />
        self.LEMMATIZATION = False<br />
        self.TF_LIST = "tf_list_stem.json" // Change the file name<br />
        self.IDF_DICT  = "idf_dict_stem.json" // Change the file name<br />
        self.INVERSE_MAPPING = "inverse_mapping_stem.json" // Change the file name<br />
        self.DOC_TITLE_LIST = "doc_title_list_stem.json"  // Change the file name<br />
        self.FILENAME = "wiki_00"<br />

run config.py:<br />
**PRE-CONDITION:**<br />
File "wiki_00" or any other file should be present in the directory<br />

**POST-CONDITION:**<br />
A file "config.pkl" should be made in the directory<br />

**2. run Corpus_Preprocessing.py**<br />
**PRECONDITION:** <br />
A file "config.pkl" should be present in the directory<br />

**POST-CONDITION:**<br />
4 json files by the name specified in "config.py" should be made in the directory. <br />


**3. run test_queries.py**<br />
**PRECONDITION:**<br />
If you want to update the query, update the "query" variable in main function. <br />
config.pkl and 4 other json files by the name specified in "config.py" should be present in the directory. <br />

**POSTCONDITION:**<br />
You must have received top 10(or less than top 10 if the query is relevant to less than 10 documents) documents for the query on console. <br />



To run for Lemmatization + Spell Check implementation: <br />

Please note that this may take more time than the previous two implementations, as lemmatization involves dictionary lookup.<br /> 

Modify it as follows:<br />
        self.STEMMING = False<br />
        self.LEMMATIZATION = True<br />
        self.TF_LIST = "tf_list_lem.json" // Change the file name<br />
        self.IDF_DICT  = "idf_dict_lem.json" // Change the file name<br />
        self.INVERSE_MAPPING = "inverse_mapping_lem.json" // Change the file name<br />
        self.DOC_TITLE_LIST = "doc_title_list_lem.json"  // Change the file name<br />
        self.FILENAME = "wiki_00"<br />

Other steps are same as the last implementation. <br />


### Dataset Link:<br />

https://drive.google.com/drive/folders/1ZsnuEm7_N6aUwhjFpv-TZXFt4DiYex4t
