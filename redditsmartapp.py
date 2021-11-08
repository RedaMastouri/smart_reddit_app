#!/usr/bin/env python
# coding: utf-8

# # Examination of Digital Community Conversations Within Specific Disease States Via Reddit

# - **Vision**: Development of a repeatable process for the analysis of Reddit conversations
# within specific condition and/or disease state with applicable threads and subreddit
# threads (subreddits) to potentially inform strategy and content development. Create a
# simplified and repeatable process that does not require the users to be fluent in Reddit.
# - **Issue**: While Reddit offers robust, open, and community-minded discussions surrounding
# conditions and disease states, Reddit also provides volumes of unstructured and
# unclassified data. The development of a repeatable process – that continues to monitor
# evolving conversations over time – currently requires multiple tools (ex. – tools to scrape
# threads, tools to analyze keyword content, tools to analyze sentiment, etc.).
# - **Method**: After identifying priority conditions and/or disease states with active Reddit
# communities (ex. – prostate cancer, breast cancer, HIV, etc.), build relational taxonomy
# (ex. – medicine, treatment, and adherence all have specific topics but have relational
# discussions) of topical themes addressed within.
# - **Potential Output**: Provide use case for healthcare companies on the importance of
# Reddit as an early source of social indicator of trends and conversational “lexicon” to be
# used for patient communications and programs.

# Contents
# 
# 1. Introduction
# 2. Types of Text Summarization
# 3. Text Summarization using Gensim
# 4. Text Summarization with sumy
# * LexRank
# * LSA (Latent Semantic Analysis )
# * Luhn
# * KL-Sum
# 5. What is Abstractive Text Summarization
# 5. T5 Transformers for Text Summarization
# 6. BART Transformers for Text Summarization
# 7. GPT-2 Transformers for Text Summarization
# 8. XLM Transformers for Text Summarization

# # <font color='Green'><center>Milestone 4: Web App</font>

# In[1]:


from IPython.display import Image
Image(filename='img/logo.png')


# In[ ]:


#Math
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# Streamlit App: Core Packages
import streamlit as st
import os

#Sentiment Analysis: NLP Packages
from textblob import TextBlob
from gensim.summarization.summarizer import summarize
import nltk
nltk.download('punkt')


#NER Imports
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.tokenize import sent_tokenize

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords

#Visualization
from matplotlib import pyplot as plt

#from dataset_milestone1 import df
#from cancer_dataset import cancer as df


# In[ ]:


#Headings for Web Application
st.title("Natural Language Processing Web Application Example")
st.subheader("What type of NLP service would you like to use?")

#Picking what NLP task you want to do
option = st.selectbox('NLP Service',('Sentiment Analysis', 'Entity Extraction', 'Text Summarization')) #option is stored in this variable

#Textbox for text user is entering
st.subheader("Enter the text you'd like to analyze.")
text = st.text_input('Enter text') #text is stored in this variable

#Display results of the NLP task
st.header("Results")

#Function to take in dictionary of entities, type of entity, and returns specific entities of specific type
def entRecognizer(entDict, typeEnt):
    entList = [ent for ent in entDict if entDict[ent] == typeEnt]
    return entList



# In[ ]:


#Sentiment Analysis
if option == 'Sentiment Analysis':
    
    #Creating graph for sentiment across each sentence in the text inputted
    sents = sent_tokenize(text) #tokenizing the text data into a list of sentences
    entireText = TextBlob(text) #storing the entire text in one string
    sentScores = [] #storing sentences in a list to plot
    for sent in sents:
        text = TextBlob(sent) #sentiment for each sentence
        score = text.sentiment[0] #extracting polarity of each sentence
        sentScores.append(score) 

    #Plotting sentiment scores per sentencein line graph
    st.line_chart(sentScores) #using line_chart st call to plot polarity for each sentence
    

    #Polarity and Subjectivity of the entire text inputted
    sentimentTotal = entireText.sentiment
    st.write("The sentiment of the overall text below.")
    st.write(sentimentTotal)
    
    

#Named Entity Recognition
elif option == 'Entity Extraction':

    #Getting Entity and type of Entity
    entities = [] #list for all entities
    entityLabels = [] #list for type of entities
    doc = nlp(text) #this call extracts all entities, make sure the spacy en library is loaded
    #iterate through all entities
    for ent in doc.ents:
        entities.append(ent.text)
        entityLabels.append(ent.label_)
    entDict = dict(zip(entities, entityLabels)) #Creating dictionary with entity and entity types
    
    
    
    #Using function to create lists of entities of each type
    entOrg = entRecognizer(entDict, "ORG")
    entCardinal = entRecognizer(entDict, "CARDINAL")
    entPerson = entRecognizer(entDict, "PERSON")
    entDate = entRecognizer(entDict, "DATE")
    entGPE = entRecognizer(entDict, "GPE")

    #Displaying entities of each type
    st.write("Organization Entities: " + str(entOrg))
    st.write("Cardinal Entities: " + str(entCardinal))
    st.write("Personal Entities: " + str(entPerson))
    st.write("Date Entities: " + str(entDate))
    st.write("GPE Entities: " + str(entGPE))

#Text Summarization
else:
    summWords = summarize(text)
    st.subheader("Summary")
    st.write(summWords)


# In[ ]:




