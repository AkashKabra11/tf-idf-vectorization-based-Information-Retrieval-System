#!/usr/bin/env python
# coding: utf-8

# In[62]:


from bs4 import BeautifulSoup
import nltk
import math
import operator
import stringdist
import string
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import json
import pickle


# In[89]:


import io, os, sys, types
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell


# In[90]:




# In[4]:


def get_clean_corpus(FILENAME):
    """
        This method extracts all documents form the HTML file. 
        
    """
    text = open(FILENAME)
    corpus = text.readlines()
    clean_corpus = []
    doc_no = 0
    prev_doc_no = 0
    doctype_flag = 0
    doctype_end_flag = 0
    doc_list = []
    punct = set(string.punctuation)
    tag = 0
    for i in range(len(corpus)):
        stmt = ""
        flag = 1
        sz = len(corpus[i])


        for j in range(sz-1):
            if(corpus[i][j] == '<'):
                tag = 1
            if(corpus[i][j] == '<' and j+1 < sz and corpus[i][j+1] == 'd'):
                doctype_flag = 1
            if(corpus[i][j] == 'c' and j+1 < sz and corpus[i][j+1] == '>'):
                doctype_end_flag = 1
            if(doctype_flag == 0 and doctype_end_flag == 0 and tag == 0):
                stmt = stmt + (corpus[i][j])
            if(corpus[i][j] == '>'):
                tag = 0
            if(doctype_flag == 1 and corpus[i][j] == '>'):
                doctype_flag = 0
            if(doctype_end_flag == 1 and corpus[i][j] == '>'):
                doctype_end_flag = 0
                doc_no = doc_no + 1
                break

        if(stmt != ""):
            if(doc_no != prev_doc_no):
                if(len(doc_list) != 0):
                    clean_corpus.append(doc_list)
                doc_list = []
                prev_doc_no = doc_no
            stmt = stmt.lower()
            doc_list.append(stmt)
    return clean_corpus


# In[5]:


def get_doc_title_list(clean_corpus):
    """
        This method extracts the document list from a clean corpus
    """
    doc_title_list = []
    for content in clean_corpus:
        doc_title_list.append(content[0])
    num_docs = len(doc_title_list)
    return doc_title_list , num_docs


# In[6]:

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None


# In[7]:


def lemmatize(word_list):
    tagged_word_list = nltk.pos_tag(word_list)
    lw = WordNetLemmatizer()
    lemmatized_words = []
    for word in tagged_word_list:
        tag = penn_to_wn(word[1])
        if(tag is None):
            continue
        lemmatized_words.append(lw.lemmatize(word[0],tag))
    return lemmatized_words
    

def get_normalized_term_freq_list(clean_corpus, num_docs, C):
    """
        This methods returns normalized tf vectors form clean corpus
        If user has specified Stemming or Lemmatization to be done, that is also covered in this method
    """
    tf_list = []
    normalized_tf_list = []
    if C.STEMMING == True:
        ps =PorterStemmer()
    # if C.LEMMATIZATION == True:
    #     wordnet_lemmatizer = lemmatize([token])[0]
    for document in clean_corpus:
        my_dict = {}
        for term in document:
            for token in nltk.word_tokenize(term):
                if C.LEMMATIZATION == True:
                    token1 = lemmatize([token])
                    if len(token1) == 0: 
                        continue
                    else:
                        token = token1[0]
                if C.STEMMING == True:
                    token = ps.stem(token)
                if token not in my_dict.keys():
                    my_dict[token] = 0
                my_dict[token] = my_dict[token] + 1
        # This tf_list contains term frequency in a document
        tf_list.append(my_dict)
        normalized_tf_list.append(my_dict)

    # Normalize and take log
    for i in range(num_docs):
        sq = 0
        for key in tf_list[i].keys():
            normalized_tf_list[i][key] = 1 + math.log2(tf_list[i][key])
            sq = sq + (normalized_tf_list[i][key] ** 2)
        sq = math.sqrt(sq)
        for key in tf_list[i].keys():
            normalized_tf_list[i][key] = normalized_tf_list[i][key]/sq
    return normalized_tf_list


# In[7]:


def get_idf_dict(clean_corpus, tf_list, num_docs):
    """
        This method returns the idf-dictionary from the clean corpus
    """
    
    idf_dict = {}
    for i in range(num_docs):
        for key in tf_list[i].keys():
            if key not in idf_dict.keys():
                idf_dict[key] = 0
            idf_dict[key] = idf_dict[key] + 1
    
    for key in idf_dict.keys():
        idf_dict[key] = math.log2(num_docs/idf_dict[key])
    # idf_dict's keys -> all unique tokens in the corpus 
    return idf_dict


# In[8]:


def get_inverse_mapping(tf_list, idf_dict, num_docs):
    # All unique keys
    """
    This methods makes inverse mapping list using the tf and idf lists
    """
    inverse_mapping = {}
    for key in idf_dict.keys():
        doc_list = [] # Contains list of docs which contain that term with tf scores
        for i in range(num_docs):
            if key in tf_list[i].keys():
                doc_list.append((i, tf_list[i][key]))
        inverse_mapping[key] = doc_list
    return inverse_mapping


# In[9]:


def store(my_dict, filename):
    """
    Auxiliary method to store all intermediate data structures to be used by test_queries.py
    """
    my_json = json.dumps(my_dict)
    f = open(filename,"w")
    f.write(my_json)
    f.close()


# In[79]:


import dill
def get_config():
    with open('config.pkl', 'rb') as f:
        C = dill.load(f)
    return C


# In[80]:


def print_config(C):
    """
    Auxiliary method to print user specified configurations. 
    """
    print("Corpus Preprocessing would be done for these Configuations:")
    if(C.STEMMING == True):
        print("Corpus tokens would be Stemmed")
    else:
        print("NO STEMMING on corpus")
    if(C.LEMMATIZATION == True):
        print("Corpus tokens would be Lemmatized")
    else:
        print("NO LEMMATIZATION on corpus")
    print("Term Frequency list would be stored in ", C.TF_LIST)
    print("Inverse Document Frequency would be stored in ", C.IDF_DICT)
    print("Inverse Mapping would be stored in ", C.INVERSE_MAPPING)
    print("Extracted Document title list would be stored in ", C.DOC_TITLE_LIST)
    print("")


# In[87]:


def main():
    
    C = get_config()
    
    print_config(C)
    
    clean_corpus = get_clean_corpus(C.FILENAME)
    print("Finished making a clean corpus")
    
    doc_title_list, num_docs = get_doc_title_list(clean_corpus)
    print("Finished making a document title list")
    
    tf_list = get_normalized_term_freq_list(clean_corpus, num_docs, C)
    print("Finished making a term frequency List")
    
    idf_dict = get_idf_dict(clean_corpus, tf_list, num_docs)
    print("Finished making a idf dictionary")
    
    inverse_mapping = get_inverse_mapping(tf_list, idf_dict, num_docs)
    print("Finished making inverse mapping")
    
    print("Now Storing all Preprocessings")
    
    store(tf_list, C.TF_LIST)
    store(idf_dict, C.IDF_DICT)
    store(inverse_mapping, C.INVERSE_MAPPING)
    store(doc_title_list, C.DOC_TITLE_LIST)
    
    print("Corpus Preprocessing Complete!")


# In[88]:


if __name__ == "__main__":
    main()


# In[ ]:


# In[ ]:




