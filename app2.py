#!/usr/bin/env python
# coding: utf-8

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
#from streamlit import components
import streamlit.components.v1 as components
import os
from PIL import Image 
import warnings
warnings.filterwarnings("ignore")

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
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel

# Plotting tools
import pyLDAvis
#import pyLDAvis.gensim 
import pyLDAvis.gensim_models

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


#from dataset_milestone1 import datasets: Add all the different diseases 
import pandas as pd

Cerebrovascular= ''
cancer = pd.read_csv('dataset/cancer.csv')
HIV = pd.read_csv('dataset/HIV_dataset.csv')
ProstateCancer = pd.read_csv('dataset/Prostate_cancer_dataset.csv')
heart = pd.read_csv('dataset/Prostate_cancer_dataset.csv')




#Starting from the top
st.markdown("# PatientComâ„¢")
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
unclassified data. The development of a repeatable process â€“ that continues to monitor
evolving conversations over time â€“ currently requires multiple tools (ex. â€“ tools to scrape
threads, tools to analyze keyword content, tools to analyze sentiment, etc.).
- **Method**: After identifying priority conditions and/or disease states with active Reddit
communities (ex. â€“ prostate cancer, breast cancer, HIV, etc.), build relational taxonomy
(ex. â€“ medicine, treatment, and adherence all have specific topics but have relational
discussions) of topical themes addressed within.
- **Potential Output**: Provide use case for healthcare companies on the importance of
Reddit as an early source of social indicator of trends and conversational â€œlexiconâ€? to be
used for patient communications and programs.
''')
st.markdown("The data presented is of 5 different diseases - **Cancer, ProstateCancer, HIV, heart disease and cerebrovascular disease,** collected from PRAW API **https://praw.readthedocs.io/**")

if st.button("Learn more about Reda Mastouri and Kalyani Pavuluri"):
    reda=Image.open('img/mastouri.png')
    kalyani=Image.open('img/kalyani.png')
    st.markdown('''**Reda Mastouri ** Reda Mastouri is Security Data Scientist with a passion for teaching and coaching. | Data Analytics | Machine Learning | Predictive Modeling | Data Visualization | NLP | Network Analytics | Network Security | Ethical Hacking |
He is knowledgeable and technically certified engineer with 7 years of continued hands-on experience in the implementation, administration and troubleshooting..''')
    st.image(reda,width=200, caption="Reda Mastouri ðŸ¤µâ€?")
    
    st.markdown('''<br>**Reda Mastouri ** Reda Mastouri is Security Data Scientist with a passion for teaching and coaching. | Data Analytics | Machine Learning | Predictive Modeling | Data Visualization | NLP | Network Analytics | Network Security | Ethical Hacking |
He is knowledgeable and technically certified engineer with 7 years of continued hands-on experience in the implementation, administration and troubleshooting..''')
    st.image(kalyani,width=200, caption="Kalyani Pavuluri ðŸ‘©â€?ðŸ’¼â€?")
    
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
def mywordcloud(dataframe):
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import warnings
    warnings.filterwarnings("ignore")
    from collections import Counter
    c = Counter()    
    
    plt.figure(figsize = (20,20))
    W_C = WordCloud(min_font_size=3, max_words=3200, width=1600, height=850, stopwords=STOPWORDS).generate(str(" ".join(dataframe.title)))
    #return plt.imshow(W_C, interpolation='bilinear')
    return W_C


#LDA
def ldavisualizer(dataset):
    #librairies
    import warnings
    warnings.filterwarnings("ignore")
    # Run in python console
    import nltk; 
    nltk.download('stopwords')

    # Gensim
    '''
    NLP Librairies
    '''
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel

    # spacy for lemmatization
    import spacy

    # Plotting tools
    import pyLDAvis
    #import pyLDAvis.gensim 
    import pyLDAvis.gensim_models
    import matplotlib.pyplot as plt

    # Enable logging for gensim - optional
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    
    #Body pkgs
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    
    #wordings
    df_words = list(sent_to_words(dataset))
    
    # Build the bigram and trigram models

    bigram = gensim.models.Phrases(df_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[df_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    

    
    #remove stop words
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    # NLTK Stop words

    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use','a','about', 'above', 'across'])
    
    st1= ['after', 'afterwards','again','against', 'all', 'almost','alone','along',
           'already',
           'also',
           'although',
           'always',
           'am',
           'among',
           'amongst',
           'amoungst',
           'amount',
           'an',
           'and',
           'another',
           'any',
           'anyhow',
           'anyone',
           'anything',
           'anyway',
           'anywhere',
           'are',
           'around',
           'as',
           'at',
           'back',
           'be',
           'became',
           'because',
           'become',
           'becomes',
           'becoming',
           'been',
           'before',
           'beforehand',
           'behind',
           'being',
           'below',
           'beside',
           'besides',
           'between',
           'beyond',
           'bill',
           'both',
           'bottom',
           'but',
           'by',
           'call',
           'can',
           'cannot',
           'cant',
           'co',
           'con',
           'could',
           'couldnt',
           'cry',
           'de',
           'describe',
           'detail',
           'do',
           'done',
           'down',
           'due',
           'during',
           'each',
           'eg',
           'eight',
           'either',
           'eleven',
           'else',
           'elsewhere',
           'empty',
           'enough',
           'etc',
           'even',
           'ever',
           'every',
           'everyone',
           'everything',
           'everywhere',
           'except',
           'few',
           'fifteen',
           'fifty',
           'fill',
           'find',
           'fire',
           'first',
           'five',
           'for',
           'former',
           'formerly',
           'forty',
           'found',
           'four',
           'from',
           'front',
           'full',
           'further',
           'get',
           'give',
           'go',
           'had',
           'has',
           'hasnt',
           'have',
           'he',
           'hence',
           'her',
           'here',
           'hereafter',
           'hereby',
           'herein',
           'hereupon',
           'hers',
           'herself',
           'him',
           'himself',
           'his',
           'how',
           'however',
           'hundred',
           'i',
           'ie',
           'if',
           'in',
           'inc',
           'indeed',
           'interest',
           'into',
           'is',
           'it',
           'its',
           'itself',
           'keep',
           'last',
           'latter',
           'latterly',
           'least',
           'less',
           'ltd',
           'made',
           'many',
           'may',
           'me',
           'meanwhile',
           'might',
           'mill',
           'mine',
           'more',
           'moreover',
           'most',
           'mostly',
           'move',
           'much',
           'must',
           'my',
           'myself',
           'name',
           'namely',
           'neither',
           'never',
           'nevertheless',
           'next',
           'nine',
           'no',
           'nobody',
           'none',
           'noone',
           'nor',
           'not',
           'nothing',
           'now',
           'nowhere',
           'of',
           'off',
           'often',
           'on',
           'once',
           'one',
           'only',
           'onto',
           'or',
           'other',
           'others',
           'otherwise',
           'our',
           'ours',
           'ourselves',
           'out',
           'over',
           'own',
           'part',
           'per',
           'perhaps',
           'please',
           'put',
           'rather',
           're',
           'same',
           'see',
           'seem',
           'seemed',
           'seeming',
           'seems',
           'serious',
           'several',
           'she',
           'should',
           'show',
           'side',
           'since',
           'sincere',
           'six',
           'sixty',
           'so',
           'some',
           'somehow',
           'someone',
           'something',
           'sometime',
           'sometimes',
           'somewhere',
           'still',
           'such',
           'system',
           'take',
           'ten',
           'than',
           'that',
           'the',
           'their',
           'them',
           'themselves',
           'then',
           'thence',
           'there',
           'thereafter',
           'thereby',
           'therefore',
           'therein',
           'thereupon',
           'these',
           'they',
           'thick',
           'thin',
           'third',
           'this',
           'those',
           'though',
           'three',
           'through',
           'throughout',
           'thru',
           'thus',
           'to',
           'together',
           'too',
           'top',
           'toward',
           'towards',
           'twelve',
           'twenty',
           'two',
           'un',
           'under',
           'until',
           'up',
           'upon',
           'us',
           'very',
           'via',
           'was',
           'we',
           'well',
           'were',
           'what',
           'whatever',
           'when',
           'whence',
           'whenever',
           'where',
           'whereafter',
           'whereas',
           'whereby',
           'wherein',
           'whereupon',
           'wherever',
           'whether',
           'which',
           'while',
           'whither',
           'who',
           'whoever',
           'whole',
           'whom',
           'whose',
           'why',
           'will',
           'with',
           'within',
           'without',
           'would',
           'yet',
           'you',
           'your',
           'yours',
           'yourself',
           'yourselves']
    
    stop_words.extend(st1)
    # Remove Stop Words
    data_words_nostops = remove_stopwords(df_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
   # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    
    # Compute Perplexity

    #print('\nPerplexity: ', lda_model.log_perplexity(corpus)) 

    # Compute Coherence Score

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    #print('\nCoherence Score: ', coherence_lda)
    
    pyLDAvis.enable_notebook()
    panel = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds='tsne', sort_topics=True)
    #topic_data =  pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds = 'tsne', sort_topics=True)
    return panel



#Scattertext

def scattertextplot(convention_df):
    #librairies
    import scattertext as st
    import re, io
    from pprint import pprint
    import pandas as pd
    import numpy as np
    import spacy
    from scipy.stats import rankdata, hmean, norm
    import os, pkgutil, json, urllib
    from urllib.request import urlopen
    from IPython.display import IFrame
    from IPython.core.display import display, HTML
    from scattertext import CorpusFromPandas, produce_scattertext_explorer
    display(HTML("<style>.container { height:100%!important; width:100% !important; }</style>"))
    
    #NLP
    from spacy.lang.en import English

    raw_text = 'Hello, world. Here are two sentences.'
    nlp = English()
    nlp.add_pipe('sentencizer')
    doc = nlp(raw_text)
    
    convention_df.groupby('comments').apply(lambda x: x.comments.apply(lambda x: len(x.split())).sum())
    convention_df['parsed'] = convention_df.title.apply(nlp)
    
    #Corpus
    corpus = st.CorpusFromParsedDocuments(convention_df, category_col='title', parsed_col='parsed').build()
    
    #Stats
    term_freq_df = corpus.get_term_freq_df()
    term_freq_df['cure_precision'] = term_freq_df['CANCER FREE!! freq'] * 1./(term_freq_df['Itâ€™s over freq'] + term_freq_df['Goodbye my sweet angel. I Lost my 5 year old daughter last night to complications from the treatment for stage IV alveolar rhabdomyosarcoma. No more tubes, no more pokes, no more drugs making her feel sick. No more pain. freq'])
    term_freq_df['cure_recall'] = term_freq_df['Officially 12 months cancer free freq'] * 1./term_freq_df['My initial prognosis was, "a few months". I recently celebrated my 3rd post diagnosis birthday! freq'].sum()
    term_freq_df['cure_f_score'] = term_freq_df.apply(lambda x: (hmean([x['cure_precision'], x['cure_recall']])
                                                                       if x['cure_precision'] > 0 and x['cure_recall'] > 0 
                                                                       else 0), axis=1)     
    #precision and recall
    term_freq_df['cure_precision_pctl'] = rankdata(term_freq_df['cure_precision'])*1./len(term_freq_df)
    term_freq_df['cure_recall_pctl'] = rankdata(term_freq_df['cure_recall'])*1./len(term_freq_df)
    
    #Normalizing 
    def normcdf(x):
        return norm.cdf(x, x.mean(), x.std())
    
    #calc
    term_freq_df['cure_precision_normcdf'] = normcdf(term_freq_df['cure_precision'])
    term_freq_df['cure_recall_normcdf'] = normcdf(term_freq_df['cure_recall'])
    
    #Override
    term_freq_df['cure_precision_normcdf'].fillna(5)
    
    #Cure
    term_freq_df['dem_corner_score'] = corpus.get_rudder_scores('cure')
    
    #HTML
    html = produce_scattertext_explorer(corpus,
                                        category='cure',
                                        category_name='cure',
                                        not_category_name='disease',
                                        width_in_pixels=1000,
                                        minimum_term_frequency=5,

                                        pmi_filter_thresold=4,
                                        metadata=convention_df['comments'],
                                        term_significance = st.LogOddsRatioUninformativeDirichletPrior())
    file_name = 'dataset/diseaseScatterWording.html'
    open(file_name, 'wb').write(html.encode('utf-8'))
    page = IFrame(src=file_name, width = 1200, height=2700)
    
    return page
    
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
In an attempt to build an AI-ready workforce, Microsoft announced Intelligent Cloud Hub which has been launched to empower the next generation of students with AI-ready skills. Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. As part of the program, the Redmond giant which wants to expand its reach and is planning to build a strong developer ecosystem in India with the program will set up the core AI infrastructure and IoT Hub for the selected campuses. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning.According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, "With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset. This will require more collaborations and training and working with AI. Thatâ€™s why it has become more critical than ever for educational institutions to integrate new cloud and AI technologies. The program is an attempt to ramp up the institutional set-up and build capabilities among the educators to educate the workforce of tomorrow." The program aims to build up the cognitive skills and in-depth understanding of developing intelligent cloud connected solutions for applications across industry. Earlier in April this year, the company announced Microsoft Professional Program In AI as a learning track open to the public. The program was developed to provide job ready skills to programmers who wanted to hone their skills in AI and data science with a series of online courses which featured hands-on labs and expert instructors as well. This program also included developer-focused AI school that provided a bunch of assets to help build AI skills.

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
	st.subheader("I - ðŸ“Š Visualization:")
	st.markdown('''
    For visualization, click any of the checkboxes to get started.
    ''')   
	if st.checkbox("Preview the WorldCloud of your sub datasets"):
		st.subheader("WorldCloud visualization ..")

		summary_options = st.selectbox("Choose dataset:",['Cancer','ProstateCancer', 'HIV', 'heart disease', 'Cerebrovascular disease'])
		if st.button("Preview"):
			if summary_options == 'Cancer':
				summary_result = mywordcloud(cancer)
				st.set_option('deprecation.showPyplotGlobalUse', False)
				plt.imshow(summary_result, interpolation='bilinear')
				plt.axis("off")
				plt.show()
				st.pyplot()
			elif summary_options == 'ProstateCancer':
				summary_result = mywordcloud(ProstateCancer)
				st.set_option('deprecation.showPyplotGlobalUse', False)
				plt.imshow(summary_result, interpolation='bilinear')
				plt.axis("off")
				plt.show()
				st.pyplot()
			elif summary_options == 'HIV':
				summary_result = mywordcloud(HIV)
				st.set_option('deprecation.showPyplotGlobalUse', False)
				plt.imshow(summary_result, interpolation='bilinear')
				plt.axis("off")
				plt.show()
				st.pyplot()
			elif summary_options == 'heart disease':
				summary_result = mywordcloud(heart)
				st.set_option('deprecation.showPyplotGlobalUse', False)
				plt.imshow(summary_result, interpolation='bilinear')
				plt.axis("off")
				plt.show()
				st.pyplot()
			elif summary_options == 'Cerebrovascular disease':
				summary_result = mywordcloud(Cerebrovascular)
				st.set_option('deprecation.showPyplotGlobalUse', False)
				plt.imshow(summary_result, interpolation='bilinear')
				plt.axis("off")
				plt.show()
				st.pyplot()
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Cancer Dataset ..")
				summary_result = mywordcloud(cancer)
				st.set_option('deprecation.showPyplotGlobalUse', False)
				plt.imshow(summary_result, interpolation='bilinear')
				plt.axis("off")
				plt.show()
				st.pyplot()
			st.success(summary_result)
    
	if st.checkbox("Preview the Latent Dirichlet Allocation (LDA) topics graphs per datasets .."):
		st.subheader("Topics visualization ..")

		summary_options = st.selectbox("Pick a dataset:",['Cancer','ProstateCancer', 'HIV', 'heart disease', 'Cerebrovascular disease'])
		if st.button("Showcase now"):
			if summary_options == 'Cancer':
				panel = ldavisualizer(cancer.comments)
				#st.set_option('deprecation.showPyplotGlobalUse', False)
				#pyLDAvis.display(panel)
				#html_string = pyLDAvis.prepared_data_to_html(prepared_pyLDAvis_data)
				#components.v1.html(diplo_string, width=1300, height=800, scrolling=True)
                				
				summary_result = pyLDAvis.display(panel)
				#plt.imshow(summary_result)
				#plt.axis("off")
				#plt.show()
				#st.pyplot()
			elif summary_options == 'ProstateCancer':
				summary_result = mywordcloud(ProstateCancer)
			elif summary_options == 'HIV':
				summary_result = mywordcloud(HIV)
			elif summary_options == 'heart disease':
				summary_result = mywordcloud(heart)
			elif summary_options == 'Cerebrovascular disease':
				summary_result = mywordcloud(Cerebrovascular)
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Cancer Dataset ..")
				summary_result = summarize(message)
			st.success(summary_result)    

	if st.checkbox("Preview the ScatterText per datasets .."):
		st.subheader("Scatter Keywords per Comments --  visualization ..")

		summary_options = st.selectbox("Search a dataset:",['Cancer','ProstateCancer', 'HIV', 'heart disease', 'Cerebrovascular disease'])
		if st.button("Give it a try"):
			if summary_options == 'Cancer':
				page = scattertextplot(cancer)
				summary_result = page
				st.text("Voila .. ")

				HtmlFile = open("dataset/diseaseScatterWording.html", 'r', encoding='utf-8')
				source_code = HtmlFile.read() 
				#print(source_code)
				components.html(source_code)
               
				                
				#plt.imshow(summary_result)
				#plt.axis("off")
				#plt.show()
				#st.pyplot()
			elif summary_options == 'ProstateCancer':
				summary_result = mywordcloud(ProstateCancer)
			elif summary_options == 'HIV':
				summary_result = mywordcloud(HIV)
			elif summary_options == 'heart disease':
				summary_result = mywordcloud(heart)
			elif summary_options == 'Cerebrovascular disease':
				summary_result = mywordcloud(Cerebrovascular)
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Cancer Dataset ..")
				summary_result = summarize(message)
			st.success(summary_result)   
    
	st.subheader("II - ðŸ§ª Advanced NLP ML:")
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
		risala = st.text_area("Type a text",placeholder)
		sents = sent_tokenize(risala) #tokenizing the text data into a list of sentences
		entireText = TextBlob(risala) #storing the entire text in one string
		sentScores = [] #storing sentences in a list to plot
		for sent in sents:
			memo = TextBlob(sent) #sentiment for each sentence
			score = memo.sentiment[0] #extracting polarity of each sentence
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

	# Comment Generation
	st.subheader("III - ðŸ”¬ Comment Generation:")
	st.markdown('''
    For comment generation, based on the subjectivity reslting from sentiment analysis, click any of the checkboxes to get started.
    ''')      
	if st.checkbox("Click here to select the reddit topic:"):
		message_to_gen = st.text_area("Enter Text","Type something ..")
		st.text("Generated comment is:")
		summary_result = gptSummarizer(message_to_gen)
		st.success(summary_result)        
        
	# Sidebar
	st.sidebar.subheader("About the App")
	logobottom=Image.open('img/logo.png')
	st.sidebar.image(logobottom,width=150)
	st.sidebar.text("PatientCom via REDDIT ðŸ¤–")
	st.sidebar.info("Examination of Digital Community Conversations Within Specific Disease States Via Reddit")   
	st.sidebar.markdown("[Data Source API](https://praw.readthedocs.io/en/stable/")
	st.sidebar.info("Linkedin [Reda Mastouri](https://www.linkedin.com/in/reda-mastouri/) ")
	st.sidebar.info("Linkedin [Kalyani Pavuluri](https://www.linkedin.com/in/kalyani-pavuluri-30416519) ")
	st.sidebar.text("PationComâ„¢ - Copyright Â© 2021")




if __name__ == '__main__':
	main()


# In[ ]:




