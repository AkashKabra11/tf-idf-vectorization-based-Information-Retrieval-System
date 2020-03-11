#!/usr/bin/env python
# coding: utf-8

# In[25]:


import io, os, sys, types
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell


# In[41]:


class Config():
    def __init__(self):
        
#         Is Stemming being used
        self.STEMMING = False
        
#         Is Lemmatization being used
        self.LEMMATIZATION = False
        
#         Term Frequency list would be stored in this file
        self.TF_LIST = "tf_list.json"
        
#         Inverse Document Frequency would be stored in this file
        self.IDF_DICT  = "idf_dict.json"
        
#         Inverse Mapping would be stored in this file
        self.INVERSE_MAPPING = "inverse_mapping.json"
        
#         Extracted Document title list would be stored in this file 
        self.DOC_TITLE_LIST = "doc_title_list.json"
        
#         Input file path
        self.FILENAME = "wiki_00"


# In[42]:


import dill
with open('config.pkl', 'wb') as output:
    C = Config()
    dill.dump(C, output)

