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

#Open Ai GPT-3
import openai
openai.api_key = "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"


# NLP Pkgs
from textblob import TextBlob
import spacy
from gensim.summarization.summarizer import summarize
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
#from dataset_milestone1 import datasets: Add all the different diseases 
import pandas as pd
cancer = pd.read_csv('dataset/cancer.csv')

#Starting from the top
st.markdown("# PationCom™")
st.markdown("By Reda Mastouri & Kalyani Pavuluri")
original_title = '<p style="color:Orange; font-size: 30px;">Examination of Digital Community Conversations Within Specific Disease States Via Reddit</p>'
st.markdown(original_title, unsafe_allow_html=True)

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
st.markdown("The data presented is of 5 different diseases - **Cancer, ProstateCancer, HIV, heart disease and cerebrovascular disease,** collected from PRAW API **https://praw.readthedocs.io/**")

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

token_text = '<p style="color:red; font-size: 20px;">Since we are using a beta version of GPT-3, let\'s type it in here instead of restaging the app</p>'
st.markdown(token_text, unsafe_allow_html=True)
gpt3token = st.text_area("Type in the newest GPT-3 Token - Example: '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' ",'Add token here ..')


#GPT-3 Text summarizer
def gptSummarizer(text):
    import os
    import openai

    openai.api_key = gpt3token

    response = openai.Completion.create(
      engine="davinci-instruct-beta",
      prompt=text,
      temperature=1,
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    #A = response.get('choices')[0]
    #answer = A.get('text')
    return response

#KLSummarizer
def KLSummy(text):
    # Creating the parser
    from sumy.summarizers.kl import KLSummarizer
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    parser=PlaintextParser.from_string(text,Tokenizer('english'))
    
    # Instantiating the KLSummarizer
    kl_summarizer=KLSummarizer()
    kl_summary=kl_summarizer(parser.document,sentences_count=3)
    
    response = []
    texto = ''
    # Printing the summary
    for sentence in kl_summary:
        response.append(sentence)
    
    for i in response:
        texto = texto + str(i)
    
    return texto    
    
#LexRank
def LexRankSummarizer(text):
    # Importing the parser and tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    # Import the LexRank summarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    
    # Initializing the parser
    my_parser = PlaintextParser.from_string(text,Tokenizer('english'))
    
    # Creating a summary of 3 sentences.
    lex_rank_summarizer = LexRankSummarizer()
    lexrank_summary = lex_rank_summarizer(my_parser.document,sentences_count=3)
    
    response = []
    texto = ''
    # Printing the summary
    for sentence in lexrank_summary:
        response.append(sentence)
    
    for i in response:
        texto = texto + str(i)
    
    return texto

#Luhn
def LuhnSummy(text):
    # Importing the parser and tokenizer
    # Import the summarizer
    from sumy.summarizers.luhn import LuhnSummarizer
    
    # Creating the parser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    parser=PlaintextParser.from_string(text,Tokenizer('english'))
    
    # Creating the summarizer
    luhn_summarizer=LuhnSummarizer()
    luhn_summary=luhn_summarizer(parser.document,sentences_count=3)

    
    response = []
    texto = ''
    # Printing the summary
    for sentence in luhn_summary:
        response.append(sentence)
    
    for i in response:
        texto = texto + str(i)
    
    return texto


#Latent Semantic Analysis, LSA
def LSASummy(text):
    # Importing the parser and tokenizer
    from sumy.summarizers.lsa import LsaSummarizer

    # Parsing the text string using PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    parser=PlaintextParser.from_string(text,Tokenizer('english'))
    
    # creating the summarizer
    lsa_summarizer=LsaSummarizer()
    lsa_summary= lsa_summarizer(parser.document,3)
    
    response = []
    texto = ''
    # Printing the summary
    for sentence in lsa_summary:
        response.append(sentence)
    
    for i in response:
        texto = texto + str(i)
    
    return texto
    



#===========================================================================


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

placeholder = '''
In an attempt to build an AI-ready workforce, Microsoft announced Intelligent Cloud Hub which has been launched to empower the next generation of students with AI-ready skills. Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. As part of the program, the Redmond giant which wants to expand its reach and is planning to build a strong developer ecosystem in India with the program will set up the core AI infrastructure and IoT Hub for the selected campuses. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning.According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, "With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset. This will require more collaborations and training and working with AI. That’s why it has become more critical than ever for educational institutions to integrate new cloud and AI technologies. The program is an attempt to ramp up the institutional set-up and build capabilities among the educators to educate the workforce of tomorrow." The program aims to build up the cognitive skills and in-depth understanding of developing intelligent cloud connected solutions for applications across industry. Earlier in April this year, the company announced Microsoft Professional Program In AI as a learning track open to the public. The program was developed to provide job ready skills to programmers who wanted to hone their skills in AI and data science with a series of online courses which featured hands-on labs and expert instructors as well. This program also included developer-focused AI school that provided a bunch of assets to help build AI skills.

'''
def main():
	""" NLP Based App with Streamlit """

	# Title
	st.title("Let's get started ..")
	st.subheader("Description")
	st.markdown('''
    	+ Because Reddit is regarded as one of the most effective social network sources for tracking the prevalence of public interests in infectious diseases (e.g., Coronavirus, HIV, and cancer) and controversial health-related issues (e.g., electronic cigarettes and marijuana) over time, reporting on findings derived from social media data nowadays becomes critical for understanding public reactions to infectious diseases. 

        + As a result, we require a faster, more intelligent, and more accurate sentiment analyzer and web scrapper-based engine capable of tracking the latest trends on novel diseases, as well as any conversational "lexicon."
        
        + This will serve as a social indicator, providing a collection of use cases for healthcare companies to sensitize consumers through various mediums, communications, and programs to learn about either polemics or significant takeaways from what is happening in social media..
    	''')
	# DatSet:
	st.subheader("A quick look at the dataset:")
	st.markdown('''
    To preview the datset, please check below.
    ''')
	st.sidebar.markdown("## Side Panel")
	st.sidebar.markdown("Use this panel to explore the dataset and create own viz.")
	st.header("Now, Explore Yourself the Time Series Dataset")
	# Create a text element and let the reader know the data is loading.
	data_load_state = st.text('Loading disease dataset...')

	# Notify the reader that the data was successfully loaded.
	data_load_state.text('Loading diseases dataset...Completed!')
	bot=Image.open('img/bot.png')
	st.image(bot,width=150)   	
    # Showing the original raw data
	if st.checkbox("Show Raw Data", False):
		st.subheader('Raw data')
		st.write(cancer)
        
        
	st.title('Quick  Explore')
	st.sidebar.subheader(' Quick  Explore')
	st.markdown("Tick the box on the side panel to explore the dataset.")


	if st.sidebar.checkbox('Basic info'):
		if st.sidebar.checkbox('Quick Look'):
			st.subheader('Dataset Quick Look:')
			st.write(cancer.head())
		if st.sidebar.checkbox("Show Columns"):
			st.subheader('Show Columns List')
			all_columns = cancer.columns.to_list()
			st.write(all_columns)
       
		if st.sidebar.checkbox('Statistical Description'):
			st.subheader('Statistical Data Descripition')
			st.write(cancer.describe())
		if st.sidebar.checkbox('Missing Values?'):
			st.subheader('Missing values')
			st.write(cancer.isnull().sum())


	# Visualization:   
	st.subheader("Visualization::")
	st.markdown('''
    For visualization, click any of the checkboxes to get started.
    ''')   
    
	st.subheader("Advanced NLP ML:")
	st.markdown('''
    For NLP deep diving, click any of the checkboxes to get started.
    ''')   
	# Summarization
	if st.checkbox("Get the summary of your text"):
		st.subheader("Summarize Your Text")

		message = st.text_area("Enter Text",placeholder)
		summary_options = st.selectbox("Choose Summarizer",['GPT-3','gensim', 'KLSummarizer', 'LexRankSummarizer', 'LuhnSummy', 'Latent Semantic Analysis'])
		if st.button("Summarize"):
			if summary_options == 'GPT-3':
				st.text(placeholder)
				summary_result = gptSummarizer(message)
			elif summary_options == 'Latent Semantic Analysis':
				st.text(placeholder)
				summary_result = LSASummy(message)
			elif summary_options == 'KLSummarizer':
				st.text(placeholder)
				summary_result = KLSummy(message)
			elif summary_options == 'LexRankSummarizer':
				st.text(placeholder)
				summary_result = LexRankSummarizer(message)
			elif summary_options == 'LuhnSummy':
				st.text(placeholder)
				summary_result = LuhnSummy(message)
			elif summary_options == 'gensim':
				st.text(placeholder)
				summary_result = summarize(message)
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Gensim Summarizer ..")
				summary_result = summarize(message)
			st.success(summary_result)


	#Sentiment Analysis
	if st.checkbox("Sentiment Analysis: Get the Sentiment Score of your text"):
		#Creating graph for sentiment across each sentence in the text inputted
		text = st.text_area("Enter Text",placeholder)
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


	# Sidebar
	st.sidebar.subheader("About the App")
	logobottom=Image.open('img/logo.png')
	st.sidebar.image(logobottom,width=150)
	st.sidebar.text("PatientCom via REDDIT 🤖")
	st.sidebar.info("Examination of Digital Community Conversations Within Specific Disease States Via Reddit")   
	st.sidebar.markdown("[Data Source API](https://praw.readthedocs.io/en/stable/")
	st.sidebar.info("Linkedin [Reda Mastouri](https://www.linkedin.com/in/reda-mastouri/) ")
	st.sidebar.info("Linkedin [Kalyani Pavuluri](https://www.linkedin.com/in/kalyani-pavuluri-30416519) ")
	st.sidebar.text("PationCom™ - Copyright © 2021")




if __name__ == '__main__':
	main()


# In[ ]:




