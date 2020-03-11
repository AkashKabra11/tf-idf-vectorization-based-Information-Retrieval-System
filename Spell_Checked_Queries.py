#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import nltk
import math
import operator
import fuzzy
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

from test_queries import get_top_K

# In[40]:


def read_from_file(filename):
    """
        Auxiliary method to read from file
    """
    with open(filename) as f:
      my_dict = json.load(f)
    return my_dict



# In[33]:


def preprocess_query(query, C):
    """
    This method preprocesses the query by stemming/lemmatizing if directed. 
    It lowercases the query, removes stop words and puctuations. 
    """
    new_query = ""
    if(C.STEMMING):
        ps =PorterStemmer()

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


# In[34]:



def get_config():
    with open('config.pkl', 'rb') as f:
        C = dill.load(f)
    return C


# In[35]:


def get_soundex(token):
    """
    This method returns the soundex of every query
    """
    token = token.upper()

    soundex = ""
    soundex += token[0]

    sound_dict = {"BFPV": "1", "CGJKQSXZ":"2", "DT":"3", "L":"4", "MN":"5", "R":"6", "AEIOUHWY":"."}

    for char in token[1:]:
        for key in sound_dict.keys():
            if char in key:
                code = sound_dict[key]
                if code != soundex[-1]:
                    soundex += code

    soundex = soundex.replace(".", "")
    soundex = soundex[:4].ljust(4, "0")

    return soundex


# In[36]:


def get_soundex_dict(idf_dict):
    """
    This methods precomputes the soundex code of every unique token in the corpus. 
    """
    soundex_dict = {}
    soundex = fuzzy.Soundex(4)
    for term in idf_dict.keys():
        soundex_dict[term] = get_soundex(term)
    return soundex_dict


# In[52]:


def get_avg_idf(idf_dict):
    """
    This method returns the average idf value of the entire corpus. 
    """
    avg_idf = 0
    num_keys = 0
    avg = 0
    for key in idf_dict.keys():
        avg_idf += idf_dict[key]
        
        num_keys += 1
    avg_idf /= num_keys
    return avg_idf


# In[38]:


def get_corpus_dist_set(query, idf_dict, normalized_tf_list, soundex_dict, avg, inverse_mapping, doc_title_list, C, thresh = 0.2, K = 5, wt = 0.5):
    """
    This method uses other methods to:
    1. preprocess the query. 
    2. Matches the closest terms in the corpus according to stemming and levenstein distance. 
    3. Computes distance score by : wt*(lev_distance) + (1-wt)*soundex_distance
    4. Suggests closest K words with score <= threshold. 
    5. Removes the terms in close set whose idf < average idf across the corpus(to supress false positives) 
    6. Uses the get_top_K method to retrieve documents with highest score on the updated query
     """
    query = preprocess_query(query, C)
    corpus_tokens = []
    #close_term_dict = {}
    soundex = fuzzy.Soundex(4)
    #query = unicode(source, 'utf-8')
    query_tokens = nltk.word_tokenize(query)
    #print(query_tokens)
    for key in idf_dict.keys():
        corpus_tokens.append(key)
    #print(len(corpus_tokens))

    cnt = 0
    lev_query = ""
    for token in query_tokens:
        query_lev_dict = {}
        for term in corpus_tokens:
            query_lev_dict[term] = 0
            soundex_notation_term = soundex_dict[term]
            soundex_notation_token = get_soundex(token)
            soundex_distance = stringdist.levenshtein_norm(soundex_notation_term, soundex_notation_token)
            lev_distance = stringdist.levenshtein_norm(token,term)
            query_lev_dict[term] = wt*(lev_distance) + (1-wt)*soundex_distance
            
        sorted_d = dict(sorted(query_lev_dict.items(), key=operator.itemgetter(1),reverse=False))
        #print(sorted_d)
        k = 0
        close_terms = []
        for item in sorted_d:
            if(k == K):
                break
            if(sorted_d[item] > thresh):
                break
            close_terms.append(str(item))
            #if(item == token):
            #    break
            k = k + 1
        
        #close_term_dict[token] = close_terms
        cnt = 0
        flg = False
        for i in close_terms:
            if(i != query_tokens[cnt]):
                flg = True
            
            lev_query = lev_query + i + " "
        #print(token, close_terms)
    if(flg == True):
        print("Searching instead for " + lev_query + ":")
            
    get_top_K(lev_query, 10, normalized_tf_list, idf_dict, inverse_mapping, doc_title_list, C)   

# In[53]:


def main():
    
    C = get_config()
    normalized_tf_list = read_from_file(C.TF_LIST)
    idf_dict = read_from_file(C.IDF_DICT)
    inverse_mapping = read_from_file(C.INVERSE_MAPPING)
    doc_title_list = read_from_file(C.DOC_TITLE_LIST)
    
    query = "MTB Endia"
    soundex_dict = get_soundex_dict(idf_dict)
    avg = get_avg_idf(idf_dict)
    
    get_corpus_dist_set(query, idf_dict, normalized_tf_list, soundex_dict,avg,inverse_mapping, doc_title_list, C, 0.35, 1, 0.8)


# In[54]:


if __name__ == "__main__":
    main()





