import imp
from urllib import response
from colorama import Fore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from apiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from os.path import exists
import os
import json

#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#for word embedding
import gensim
from gensim.models import Word2Vec

class SpamPredictor():

    def __init__(self):

        self.CommentPredictor = joblib.load('CommentTextModel')
        self.UsernamePredictor = joblib.load('AuthorNameModel')
        self.Comments = []
        self.Usernames = []
        self.UserIds = []

        # Initialize the lemmatizer
        self.wl = WordNetLemmatizer()


    def loadComments(self, url, numberOffCalls):

        path = self.get_id(url)

        if not exists(f'{path}'):

            print("Downloading data...")
            os.mkdir(f'{path}')
            count = 0
            
            #build our service from api key
            service = self.build_service('Api_Key.txt')

            #make an API call using our service
            response = service.commentThreads().list(
                part='id, snippet, replies',
                maxResults=100,
                textFormat='plainText',
                order='time',
                videoId=self.get_id(url)
            ).execute()

            while response:
                
                with open(f'{path}/{count}.json', "w") as i :
                    json.dump(response, i)
                
                # check for nextPageToken, and if it exists, set response equal to the Json response
                if 'nextPageToken' in response:
                    response = service.commentThreads().list(
                        part='id, snippet, replies',
                        maxResults=100,
                        textFormat='plainText',
                        order='time',
                        videoId=self.get_id(url),
                        pageToken=response['nextPageToken']
                    ).execute()
                    count+=1
                else:
                    break
        
        self.readAllData(f'{path}', numberOffCalls)
    
    def readAllData(self, path, numberOffCalls):
        
        for i in range(0, numberOffCalls):
            
            with open(f'{path}/{i}.json', 'r') as openfile:
                # Reading from json file
                response = json.load(openfile)

                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    #comment_id = item['snippet']['topLevelComment']['id']
                    reply_count = item['snippet']['totalReplyCount']
                    #isReply = False
                    authorName = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                    
                    if 'authorChannelId' in item['snippet']['topLevelComment']['snippet']:
                        authorId =  item['snippet']['topLevelComment']['snippet']['authorChannelId']['value']
                    else:
                        authorId = None

                    #append to lists
                    self.Comments.append(comment)
                    #commentsId.append(comment_id)
                    #repliesCount.append(reply_count)
                    #isReplies.append(isReply)
                    self.Usernames.append(authorName)
                    self.UserIds.append(authorId)

                    if reply_count > 0 and 'replies' in item:
                        for reply in item['replies']['comments']:
                            reply_comment = reply['snippet']['textDisplay']
                            #reply_comment_id = reply['id']
                            #reply_reply_count = 0
                            #isReply = True
                            reply_author_name = reply['snippet']['authorDisplayName']
                            if 'authorChannelId' in reply['snippet']:
                                reply_author_id = reply['snippet']['authorChannelId']['value']
                            else:
                                reply_author_id = None

                            #append to lists
                            self.Comments.append(reply_comment)
                            #commentsId.append(reply_comment_id)
                            #repliesCount.append(reply_reply_count)
                            #isReplies.append(isReply)
                            self.Usernames.append(reply_author_name)
                            self.UserIds.append(reply_author_id)
    
    def getSuspectedSpammers(self):

        zipped = list(zip(self.Comments, self.Usernames, self.UserIds))
        df = pd.DataFrame(zipped, columns=['Comment', 'UserName', 'UserID'])

        df['Clean Comment'] = df['Comment'].apply(lambda x : self.finalpreprocess(x))
        df['Clean UserName'] = df['UserName'].apply(lambda x : self.finalpreprocess(x))

        X_test = df['Clean Comment']
        X_test_author = df['Clean UserName']
        
        tfidf_vectorizer = joblib.load('tfidf_vectorizer')

        tfidf_vectorizer_UserName = joblib.load('author_tfidf_vectorizer')

        X_vector = tfidf_vectorizer.transform(X_test)
        X_vector_author = tfidf_vectorizer_UserName.transform(X_test_author)

        y_predict = self.CommentPredictor.predict(X_vector)      
        y_prob = self.CommentPredictor.predict_proba(X_vector)[:,1]

        y_predict_author = self.UsernamePredictor.predict(X_vector_author)
        y_prob_author = self.UsernamePredictor.predict_proba(X_vector_author)[:,1]

        df['predict_prob']= y_prob
        df['Is Scam byComment']= y_predict
        df['author_predict_prob'] = y_prob_author
        df['Is Scam byAuthor'] = y_predict_author

        df['Is Scam'] = df.apply(lambda x : self.decideIsScam(x), axis=1)

        Row_list = {}
        commentsPerAuthor = {}

        for index, rows in df[df['Is Scam'] == 1].iterrows():
            # Create list for the current row
            userID = rows['UserID']
            comment = rows["Comment"]
            
            # append the list to the final list
            Row_list[userID] = comment

            if userID in commentsPerAuthor.keys():
                commentsPerAuthor[userID] += 1
            else:
                commentsPerAuthor[userID] = 1

        for key, value in commentsPerAuthor.items():
            if value == 1:
                Row_list.pop(key)

        return Row_list

    def decideIsScam(self, row):
        if row['Is Scam byComment'] == 1 or row['Is Scam byAuthor'] == 1:
            return 1
        else:
            return 0

        '''elif row['Is Scam byComment'] == 0 and row['Is Scam byAuthor'] == 1:
            return 1
        elif row['Is Scam byComment'] == 1 and row['Is Scam byAuthor'] == 0:
            return 0
        else:
            return 0'''

    #convert to lowercase, strip and remove punctuations
    def preprocess(self, text):
        text = str(text).lower() 
        text=text.strip()  
        text=re.compile('<.*?>').sub('', text) 
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
        text = re.sub('\s+', ' ', text)  
        text = re.sub(r'\[[0-9]*\]',' ',text) 
        text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d',' ',text) 
        text = re.sub(r'\s+',' ',text) 
        return text

    
    # STOPWORD REMOVAL
    def stopword(self, string):
        a= [i for i in string.split() if i not in stopwords.words('english')]
        return ' '.join(a)
    
    # This is a helper function to map NTLK position tags
    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    # Tokenize the sentence
    
    def lemmatizer(self, string):
        word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
        a=[self.wl.lemmatize(tag[0], self.get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
        return " ".join(a)

    def finalpreprocess(self, string):
        return self.lemmatizer(self.stopword(self.preprocess(string)))
    
    def get_id(self, url):
        u_pars = urlparse(url)
        quer_v = parse_qs(u_pars.query).get('v')
        if quer_v:
            return quer_v[0]
        pth = u_pars.path.split('/')
        if pth:
            return pth[-1]

    def build_service(self, filename):
        with open(filename) as f:
            key = f.readline()

        YOUTUBE_API_SERVICE_NAME = "youtube"
        YOUTUBE_API_VERSION = "v3"
        return build(YOUTUBE_API_SERVICE_NAME,
                    YOUTUBE_API_VERSION,
                    developerKey=key)