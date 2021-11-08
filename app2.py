#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Description
This is a Natural Language Processing(NLP) Based App useful for basic NLP concepts such as follows;
+ Tokenization & Lemmatization using Spacy
+ Named Entity Recognition(NER) using SpaCy
+ Sentiment Analysis using TextBlob
+ Document/Text Summarization using Gensim/Sumy
This is built with Streamlit Framework, an awesome framework for building ML and NLP tools.

Purpose
To perform basic and useful NLP task with Streamlit,Spacy,Textblob and Gensim/Sumy

"""
# Core Pkgs
import streamlit as st
import os
from PIL import Image 

#Visualization
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# NLP Pkgs
from textblob import TextBlob
import spacy
from gensim.summarization.summarizer import summarize
import nltk
nltk.download('punkt')

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
#from dataset_milestone1 import df
#from cancer_dataset import cancer as df

#DATA_URL = df
st.markdown("# PationCom™")
st.markdown("By Reda Mastouri & Kalyani Pavuluri")
st.markdown("# Examination of Digital Community Conversations Within Specific Disease States Via Reddit- By ")

img=Image.open('img/logo.png')
st.image(img,width=200)
st.markdown('''
- **Vision**: Development of a repeatable process for the analysis of Reddit conversations
within specific condition and/or disease state with applicable threads and subreddit
threads (subreddits) to potentially inform strategy and content development. Create a
simplified and repeatable process that does not require the users to be fluent in Reddit.
- **Issue**: While Reddit offers robust, open, and community-minded discussions surrounding
conditions and disease states, Reddit also provides volumes of unstructured and
unclassified data. The development of a repeatable process – that continues to monitor
evolving conversations over time – currently requires multiple tools (ex. – tools to scrape
threads, tools to analyze keyword content, tools to analyze sentiment, etc.).
- **Method**: After identifying priority conditions and/or disease states with active Reddit
communities (ex. – prostate cancer, breast cancer, HIV, etc.), build relational taxonomy
(ex. – medicine, treatment, and adherence all have specific topics but have relational
discussions) of topical themes addressed within.
- **Potential Output**: Provide use case for healthcare companies on the importance of
Reddit as an early source of social indicator of trends and conversational “lexicon” to be
used for patient communications and programs.
''')
st.markdown("The data presented are of 5 different diseases - **Cancer, ProstateCancer, HIV, heart disease and cerebrovascular disease,** collected from PRAW API **https://praw.readthedocs.io/**")

if st.button("Learn more about Reda Mastouri and Kalyani Pavuluri"):
    reda=Image.open('img/mastouri.png')
    kalyani=Image.open('img/kalyani.png')
    st.markdown('''**Reda Mastouri ** Reda Mastouri is Security Data Scientist with a passion for teaching and coaching. | Data Analytics | Machine Learning | Predictive Modeling | Data Visualization | NLP | Network Analytics | Network Security | Ethical Hacking |
He is knowledgeable and technically certified engineer with 7 years of continued hands-on experience in the implementation, administration and troubleshooting..''')
    st.image(reda,width=200, caption="Reda Mastouri 🤵‍")
    
    st.markdown('''<br>**Reda Mastouri ** Reda Mastouri is Security Data Scientist with a passion for teaching and coaching. | Data Analytics | Machine Learning | Predictive Modeling | Data Visualization | NLP | Network Analytics | Network Security | Ethical Hacking |
He is knowledgeable and technically certified engineer with 7 years of continued hands-on experience in the implementation, administration and troubleshooting..''')
    st.image(kalyani,width=200, caption="Kalyani Pavuluri 👩‍💼‍")
    
    st.markdown("The data was collected and made available by **[Reda Mastouri](https://www.linkedin.com/in/reda-mastouri/**.")
    st.markdown("and **[Kalyani Pavuluri](https://www.linkedin.com/in/kalyani-pavuluri-30416519**.")
    images=Image.open('img/presentation.png')
    st.image(images,width=700)
    #Ballons
    st.balloons()



#GPT-3 Text summarizer
def gptSummarizer(text):
    import os
    import openai

    openai.api_key = "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"

    response = openai.Completion.create(
      engine="davinci-instruct-beta",
      prompt=text,
      temperature=1,
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    A = response.get('choices')[0]
    answer = A.get('text')
    return answer


# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData

# Function For Extracting Entities
@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text,entity.label_)for entity in docx.ents]
	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
	return allData


def main():
	""" NLP Based App with Streamlit """

	# Title
	st.title("Ultimate NLP Application")
	st.subheader("Natural Language Processing for everyone")
	st.markdown("""
    	#### Description
    	+ This is a Natural Language Processing(NLP) Based App useful for basic NLP task
    	Tokenization , Lemmatization, Named Entity Recognition (NER), Sentiment Analysis, Text Summarization. Built for social good by [LekanAkin](https://github.com/lekanakin). Click any of the checkboxes to get started.
    	""")

	# Summarization
	if st.checkbox("Get the summary of your text"):
		st.subheader("Summarize Your Text")

		message = st.text_area("Enter Text","Type Here....")
		summary_options = st.selectbox("Choose Summarizer",['sumy','gensim'])
		if st.button("Summarize"):
			if summary_options == 'sumy':
				st.text("Using Sumy Summarizer ..")
				summary_result = sumy_summarizer(message)
			elif summary_options == 'GPT-3':
				st.text("Using GPT-3 Summarizer ..")
				summary_result = gptSummarizer(message)
			elif summary_options == 'gensim':
				st.text("Using Gensim Summarizer ..")
				summary_result = summarize(message)
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Gensim Summarizer ..")
				summary_result = summarize(message)
			st.success(summary_result)

	# Sentiment Analysis
	if st.checkbox("Get the Sentiment Score of your text"):
		st.subheader("Identify Sentiment in your Text")

		message = st.text_area("Enter Text","Type Here...")
		if st.button("Analyze"):
			blob = TextBlob(message)
			result_sentiment = blob.sentiment
			st.success(result_sentiment)

	# Entity Extraction
	if st.checkbox("Get the Named Entities of your text"):
		st.subheader("Identify Entities in your text")

		message = st.text_area("Enter Text","Type Here..")
		if st.button("Extract"):
			entity_result = entity_analyzer(message)
			st.json(entity_result)

	# Tokenization
	if st.checkbox("Get the Tokens and Lemma of text"):
		st.subheader("Tokenize Your Text")

		message = st.text_area("Enter Text","Type Here.")
		if st.button("Analyze"):
			nlp_result = text_analyzer(message)
			st.json(nlp_result)



	st.sidebar.subheader("About the App")
	logobottom=Image.open('img/logo.png')
	st.sidebar.image(logobottom,width=150)
	st.sidebar.text("PatientCom via REDDIT 🤖")
	st.sidebar.info("Examination of Digital Community Conversations Within Specific Disease States Via Reddit")   
	st.sidebar.markdown("[Data Source API](https://praw.readthedocs.io/en/stable/")
	st.sidebar.info("Linkedin [Reda Mastouri](https://www.linkedin.com/in/reda-mastouri/) ")
	st.sidebar.info("Linkedin [Kalyani Pavuluri](https://www.linkedin.com/in/kalyani-pavuluri-30416519) ")
	st.sidebar.info("Self Exploratory Visualization using Optimal Transport on Financial Time Series Data- Brought To you By [Jospeh Bunster](https://github.com/Joseph-Bunster)  ")
	st.sidebar.text("All Right Reserved  ©")




if __name__ == '__main__':
	main()


# In[ ]:




