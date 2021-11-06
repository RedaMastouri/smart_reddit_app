#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import praw
from praw_credential import reddit


# In[2]:


'''
Dataset creator
'''

def subreddit_scrapper(subject):
    # needed librairies 
    import pandas as pd
    import praw
    import datetime
    #Connecting to the api
    from praw_credential import reddit
    
    #defining subject
    top_post = []
    subreddit = reddit.subreddit(subject)
    for post in subreddit.top(limit=None):
        top_post.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
    
    #defining the features    
    top_posts = pd.DataFrame(top_post,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
    
    #adjusting the created_date
    top_posts['created'] = pd.to_datetime(top_posts['created'], unit='s')
    top_posts.created = top_posts.created.dt.tz_localize('UTC')
    
    #Adding UID as index
    top_posts['UID'] = range(1, 1+len(top_posts))
    
    #ignoring covid wording
    df = top_posts[top_posts["title"].str.contains("COVID|corona|Covid|coronavirus|Coronavirus|covid")==False]
    
    #Returning the dataframe:
    return df


# In[3]:


'''
Showcasing cancer dataset
'''
cancer = subreddit_scrapper('cancer')
cancer


# In[4]:


'''
comment scrapper 
'''

def reddit_scrapper(subject):
    from praw.models import MoreComments
    import numpy as np
    
    #definig df:
    df = subreddit_scrapper(subject)
    df['body'].replace('', np.nan, inplace=True)
    df.dropna(subset=['body'], inplace=True)
       
    subreddit = reddit.subreddit(subject)
    #post lists
    posts = []
    # comments in dict
    title_comment_dic = {}
    
    #iterating in all subreddits and assign to column body
    for submission in subreddit.top('all'):
        posts = []
        for top_level_comment in submission.comments:
            if isinstance(top_level_comment, MoreComments):
                continue
            posts.append(top_level_comment.body)
        postdf = pd.DataFrame(posts,columns=["body"])
        indexNames = postdf[(postdf.body == '[removed]') | (postdf.body == '[deleted]')].index
        postdf.drop(indexNames, inplace=True)
        uid = df['UID'][df['title'] == submission.title].values
        if len(uid) > 0:
            title_comment_dic[uid[0]] = postdf
    
    #Returning the comment dictionary 
    return title_comment_dic


# In[5]:


'''
Merged with comment
'''
def merger(subject):
    # first let's convert the myscrapping dictionary into pandas dataframe
    df_scrapped = pd.DataFrame.from_dict(reddit_scrapper(subject), orient='index')
    df_scrapped.columns = ['comments']
    df_scrapped.index.name = 'UID'
    
    
    return df_scrapped


# In[6]:


merger('cancer')


# # Main function

# In[13]:


'''
Main function
'''

def main(subject):
    # now defining a list of frames 
    import numpy as np
    
    #definig df:
    df_prawing = subreddit_scrapper(subject)
    df_prawing['body'].replace('', np.nan, inplace=True)
    df_prawing.dropna(subset=['body'], inplace=True)
    
    df_comments = merger(subject)
    
    #a list of frames
    frames = [df_prawing , df_comments]
    
    #joining 
    #df = pd.concat(frames) will not work!
    df = df_prawing.merge(df_comments, how = 'inner', on = ['UID'])
    
    return df


# In[14]:


cancer = main('cancer')


# In[15]:


cancer


# In[ ]:




