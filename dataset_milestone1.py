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

# # <font color='Green'><center>Mileston 1: Research and Dataset</font>

# In[1]:


from IPython.display import Image
Image(filename='img/logo.png')


# # I - DataSet: Integrating the Reddit APP: PatientCom

# In[2]:


Image(filename='img/app.jpg')


# In[3]:


'''
Needed librairies
'''
import requests
import pandas as pd
import numpy as np

'''
Data analysis and Wrangling
'''
import pandas as pd
import numpy as np
import random as rnd
from datetime import date, datetime, time, timedelta
import datetime as dt

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.ticker as mtic
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[4]:


#Final DF
# https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c
import requests
import pandas as pd
from datetime import datetime



# we use this function to convert responses to dataframes
def df_from_response(res):
    # initialize temp dataframe for batch of data in response
    df = pd.DataFrame()

    # loop through each post pulled from res and append to df
    for post in res.json()['data']['children']:
        df = df.append({
            'subreddit': post['data']['subreddit'],
            'title': post['data']['title'],
            'selftext': post['data']['selftext'],
            'upvote_ratio': post['data']['upvote_ratio'],
            'ups': post['data']['ups'],
            'downs': post['data']['downs'],
            'score': post['data']['score'],
            'link_flair_css_class': post['data']['link_flair_css_class'],
            'created_utc': datetime.fromtimestamp(post['data']['created_utc']).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'id': post['data']['id'],
            'kind': post['kind'], 
            'approved_at_utc': post['data']['approved_at_utc'],
            'subreddit': post['data']['subreddit'],
            'selftext': post['data']['selftext'],
            'saved': post['data']['saved'],
            'mod_reason_title': post['data']['mod_reason_title'],
            'gilded': post['data']['gilded'],
            'clicked': post['data']['clicked'],
            'title': post['data']['title'],
            'link_flair_richtext': post['data']['link_flair_richtext'],
            'subreddit_name_prefixed': post['data']['subreddit_name_prefixed'],
            'hidden': post['data']['hidden'],
            'pwls': post['data']['pwls'],
            'link_flair_css_class': post['data']['link_flair_css_class'],
            'downs': post['data']['downs'],
            'top_awarded_type': post['data']['top_awarded_type'],
            'hide_score': post['data']['hide_score'],
            'name': post['data']['name'],
            'quarantine': post['data']['quarantine'],
            'link_flair_text_color': post['data']['link_flair_text_color'],
            'upvote_ratio': post['data']['upvote_ratio'],
            'author_flair_background_color': post['data']['author_flair_background_color'],
            'subreddit_type': post['data']['subreddit_type'],
            'ups': post['data']['ups'],
            'total_awards_received': post['data']['total_awards_received'],
            'media_embed': post['data']['media_embed'],
            'author_flair_template_id': post['data']['author_flair_template_id'],
            'is_original_content': post['data']['is_original_content'],
            'user_reports': post['data']['user_reports'],
            'secure_media': post['data']['secure_media'],
            'is_reddit_media_domain': post['data']['is_reddit_media_domain'],
            'is_meta': post['data']['is_meta'],
            'category': post['data']['category'],
            'secure_media_embed': post['data']['secure_media_embed'],
            'link_flair_text': post['data']['link_flair_text'],
            'can_mod_post': post['data']['can_mod_post'],
            'score': post['data']['score'],
            'approved_by': post['data']['approved_by'],
            'thumbnail': post['data']['thumbnail'],
            'edited': post['data']['edited'],
            'author_flair_css_class': post['data']['author_flair_css_class'],
            'gildings': post['data']['gildings'],
            'content_categories': post['data']['content_categories'],
            'is_self': post['data']['is_self'],
            'mod_note': post['data']['mod_note'],
            'created': post['data']['created'],
            'link_flair_type': post['data']['link_flair_type'],
            'wls': post['data']['wls'],
            'removed_by_category': post['data']['removed_by_category'],
            'banned_by': post['data']['banned_by'],
            'domain': post['data']['domain'],
            'allow_live_comments': post['data']['allow_live_comments'],
            'selftext_html': post['data']['selftext_html'],
            'likes': post['data']['likes'],
            'suggested_sort': post['data']['suggested_sort'],
            'banned_at_utc': post['data']['banned_at_utc'],
            'view_count': post['data']['view_count'],
            'archived': post['data']['archived'],
            'no_follow': post['data']['no_follow'],
            'is_crosspostable': post['data']['is_crosspostable'],
            'pinned': post['data']['pinned'],
            'over_18': post['data']['over_18'],
            'all_awardings': post['data']['all_awardings'],
            'awarders': post['data']['awarders'],
            'media_only': post['data']['media_only'],
            'can_gild': post['data']['can_gild'],
            'spoiler': post['data']['spoiler'],
            'locked': post['data']['locked'],
            'author_flair_text': post['data']['author_flair_text'],
            'treatment_tags': post['data']['treatment_tags'],
            'visited': post['data']['visited'],
            'removed_by': post['data']['removed_by'],
            'num_reports': post['data']['num_reports'],
            'distinguished': post['data']['distinguished'],
            'subreddit_id': post['data']['subreddit_id'],
            'mod_reason_by': post['data']['mod_reason_by'],
            'removal_reason': post['data']['removal_reason'],
            'link_flair_background_color': post['data']['link_flair_background_color'],
            'id': post['data']['id'],
            'is_robot_indexable': post['data']['is_robot_indexable'],
            'report_reasons': post['data']['report_reasons'],
            'author': post['data']['author'],
            'discussion_type': post['data']['discussion_type'],
            'num_comments': post['data']['num_comments'],
            'send_replies': post['data']['send_replies'],
            'whitelist_status': post['data']['whitelist_status'],
            'contest_mode': post['data']['contest_mode'],
            'mod_reports': post['data']['mod_reports'],
            'author_flair_text_color': post['data']['author_flair_text_color'],
            'permalink': post['data']['permalink'],
            'parent_whitelist_status': post['data']['parent_whitelist_status'],
            'stickied': post['data']['stickied'],
            'url': post['data']['url'],
            'subreddit_subscribers': post['data']['subreddit_subscribers'],
            'created_utc': post['data']['created_utc'],
            'num_crossposts': post['data']['num_crossposts'],
            'media': post['data']['media'],
            'is_video': post['data']['is_video']
        }, ignore_index=True)

    return df


# In[5]:


# authenticate API
# note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
client_auth = requests.auth.HTTPBasicAuth('5PtLw2OXKn8K-lTmDq8WaA', 'jxi9Ob2bO6axkXP_eKKQH4tFO5t9Rg')

# here we pass our login method (password), username, and password
disease = {'grant_type': 'password',
        'username': 'raymastouri',
        'password': 'M@$t0ur1198700rq'}

# setup our header info, which gives reddit a brief description of our app
headers = {'User-Agent': 'PatientCom/0.0.1'}

# send our request for an OAuth token
res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=client_auth, data=disease, headers=headers)

# convert response to JSON and pull access_token value
token = res.json()['access_token']

# add authorization to our headers dictionary
headers = {**headers, **{'Authorization': f"bearer {token}"}}

# while the token is valid (~2 hours) we just add headers=headers to our requests
requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)


# In[6]:


# initialize dataframe and parameters for pulling data in loop
disease = pd.DataFrame()
params = {'limit': 100}


# In[7]:


# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/disease/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df.iloc[len(new_df)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    disease = disease.append(new_df, ignore_index=True)


# In[8]:


disease.head()


# # <font color='Green'><center>Merging Raw Data for Subreddits</font>

# ### What are the subreddits we have chosen?

# - disease
# - futurology
# - health
# - gravesdisease
# - Addisonsdisease
# - medecine
# - fatlogic
# - ChronicPain
# - Epidemiology
# - cancer
# - coronavisus
# - covid19
# - prostatecancer

# In[9]:


'''
Health
'''
# initialize dataframe and parameters for pulling data in loop
Health = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/Health/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df2 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df2.iloc[len(new_df2)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    Health = Health.append(new_df2, ignore_index=True)
Health.head()


# In[10]:


'''
futurology
'''
# initialize dataframe and parameters for pulling data in loop
futurology = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/futurology/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df3 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df3.iloc[len(new_df3)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    futurology = futurology.append(new_df3, ignore_index=True)
futurology.head()


# In[11]:


'''
gravesdisease
'''
# initialize dataframe and parameters for pulling data in loop
gravesdisease = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/gravesdisease/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df4 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df4.iloc[len(new_df4)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    gravesdisease = gravesdisease.append(new_df4, ignore_index=True)
gravesdisease.head()


# In[12]:


'''
Addisonsdisease
'''
# initialize dataframe and parameters for pulling data in loop
Addisonsdisease = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/Addisonsdisease/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df5 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df5.iloc[len(new_df5)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    Addisonsdisease = Addisonsdisease.append(new_df5, ignore_index=True)
Addisonsdisease.head()


# In[13]:


'''
fatlogic
'''
# initialize dataframe and parameters for pulling data in loop
fatlogic = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/fatlogic/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df6 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df6.iloc[len(new_df6)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    fatlogic = fatlogic.append(new_df6, ignore_index=True)
fatlogic.head()


# In[14]:


'''
ChronicPain
'''
# initialize dataframe and parameters for pulling data in loop
ChronicPain = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/ChronicPain/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df7 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df7.iloc[len(new_df7)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    ChronicPain = ChronicPain.append(new_df7, ignore_index=True)
ChronicPain.head()


# In[15]:


'''
Epidemiology
'''
# initialize dataframe and parameters for pulling data in loop
Epidemiology = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/Epidemiology/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df8 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df8.iloc[len(new_df8)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    Epidemiology = Epidemiology.append(new_df8, ignore_index=True)
Epidemiology.head()


# In[16]:


'''
DebateVaccines
'''
# initialize dataframe and parameters for pulling data in loop
DebateVaccines = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/DebateVaccines/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df10 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df10.iloc[len(new_df10)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    DebateVaccines = DebateVaccines.append(new_df10, ignore_index=True)
DebateVaccines.head()


# In[17]:


'''
Coronavirus
'''

# initialize dataframe and parameters for pulling data in loop
Coronavirus = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/Coronavirus/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df11 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df11.iloc[len(new_df11)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    Coronavirus = Coronavirus.append(new_df11, ignore_index=True)
Coronavirus.head()


# In[18]:


'''
cancer
'''

# initialize dataframe and parameters for pulling data in loop
cancer = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/cancer/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df12 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df12.iloc[len(new_df12)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    cancer = cancer.append(new_df12, ignore_index=True)
cancer.head()


# In[19]:


'''
ProstateCancer
'''

# initialize dataframe and parameters for pulling data in loop
ProstateCancer = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/ProstateCancer/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df13 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df13.iloc[len(new_df13)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    ProstateCancer = ProstateCancer.append(new_df13, ignore_index=True)
ProstateCancer.head()


# In[20]:


'''
COVID19
'''

# initialize dataframe and parameters for pulling data in loop
COVID19 = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/COVID19/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df14 = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df14.iloc[len(new_df14)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    COVID19 = COVID19.append(new_df14, ignore_index=True)
COVID19.head()


# # <font color='Green'><center>Aggregating all the Data in one Dataset</font>

# In[21]:


frames = [disease, futurology, Health, gravesdisease, Addisonsdisease, fatlogic, ChronicPain, Epidemiology, DebateVaccines, Coronavirus, cancer, ProstateCancer, COVID19]
df = pd.concat(frames)
df

