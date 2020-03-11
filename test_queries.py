#!/usr/bin/env python
# coding: utf-8

# In[1]:

from bs4 import BeautifulSoup
import nltk
import math
import operator
import stringdist
import string
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import json
import dill

from Corpus_Preprocessing import lemmatize, is_noun, is_verb, is_adverb, is_adjective, penn_to_wn


# In[2]:


import io, os, sys, types
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell


# In[3]:


def read_from_file(filename):
    """
        Auxiliary method to read from file
    """
    with open(filename) as f:
      my_dict = json.load(f)
    return my_dict


# In[4]:


def preprocess_query(query, C):
    new_query = ""
    if(C.STEMMING):
        ps =PorterStemmer()
    #if(C.LEMMATIZATION):
        #wordnet_lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(query)
    punct = set(string.punctuation)
    for i in tokens: 
        i = i.lower()
        if(C.LEMMATIZATION):
            i1 = lemmatize([i])
            if len(i1) == 0:
                continue
            else:
                i = i1[0]

        if(C.STEMMING):
            i = ps.stem(i)
        if(i not in punct):
            new_query = new_query + i + " "
            
    return new_query


# In[5]:


def get_query_tf_idf(query_tokens, idf_dict):
    term_freq = {}
    for key in query_tokens:
        if key not in term_freq.keys():
            term_freq[key] = 0
        term_freq[key] = term_freq[key] + 1
    sq = 0
    for key in term_freq.keys():
        if key in idf_dict.keys():
            term_freq[key] = term_freq[key]*idf_dict[key]
        else:
            term_freq[key] = 0
        sq = sq + (term_freq[key] ** 2)
        #print(term_freq[key])
        
    sq = math.sqrt(sq)
    for key in term_freq.keys():
        if(sq > 0):
            term_freq[key] = term_freq[key]/sq
    return term_freq


# In[6]:


def get_top_K(query, K, normalized_tf_list, idf_dict, inverse_mapping, doc_title_list, C):
    """
    This function gets top k documents from corpus
    """
    query = preprocess_query(query, C)
    #print(query)
    score_dict = {}
    query_tokens = nltk.word_tokenize(query)
    tf_idf_query_dict = get_query_tf_idf(query_tokens, idf_dict)
    # for i in tf_idf_query_dict.keys():
    #     print(i, tf_idf_query_dict[i])
    for i in range(len(normalized_tf_list)):
        score_dict[i] = 0
    for key in tf_idf_query_dict.keys():
        # print(key)
        if key in inverse_mapping.keys():
            # print(inverse_mapping[key])
            # inverse_mapping[key] has format [(doc_no, tf_score), ....]
            for tf_score_tuple in inverse_mapping[key]:
                score_dict[tf_score_tuple[0]] += tf_score_tuple[1]*tf_idf_query_dict[key]
                
    sorted_d = dict(sorted(score_dict.items(), key=operator.itemgetter(1),reverse=True))
    k = 0

    for i in sorted_d:
        if(k == K or sorted_d[i] == 0):
            break
        
        print("{}th Document is {} with Score {}".format(k+1,doc_title_list[i].upper(), sorted_d[i]*100))

        k = k + 1



# In[7]:



def get_config():
    with open('config.pkl', 'rb') as f:
        C = dill.load(f)
    return C

# In[9]:
def main():
    C = get_config()
    normalized_tf_list = read_from_file(C.TF_LIST)
    idf_dict = read_from_file(C.IDF_DICT)
    inverse_mapping = read_from_file(C.INVERSE_MAPPING)
    doc_title_list = read_from_file(C.DOC_TITLE_LIST)
    
    
    query = "MTV India"
    get_top_K(query, 10, normalized_tf_list, idf_dict, inverse_mapping, doc_title_list, C)


# In[10]:

if __name__ == "__main__":
    main()