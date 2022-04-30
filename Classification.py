import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

data = pd.read_csv('comments.csv', sep=',', header=0)

scamChannels = ['UCwgMHGG8IDcYjjSXtYbE9rQ', 'UC8Oy99fOvCjHfbvTImDXpkg',
'UCswOElw6g7pEeAA5Mu15p3Q', 'UCIknJ8HTOMLSfIAmtsm84vA', 'UCsXIMkerN0ofYQVIvNWr7Dg', 'UC5iFMKp-Tuf2RX8wTDIA00w',
'UC4BTXtDeOzz85XMShFWY1VA', 'UCIIab0_13sTLxQY66QjpM-g', 'UCZIIrgNdsq0MPdbG6utOLmw', 'UCf3mY9fz0oesuFS4fUnhjsg',
'UCii92DZYgGqYquT71D9cWSQ', 'UCSkoxrUobYERf5FI1F0KDTg', 'UCbdM2ysSVcPxHghp50tiPQw', 'UC1Wi-CaZ_u111EET3es-jzA',
'UCJB7ZvEbo-xRZum_3Zmq9sw', 'UCe5DcGeti6adyFedf49MRuA', 'UCfOYoX3Cwcn0A8t32c3cM5g', 'UCIAYvaVP15Cel4NZ1ib_ADA',
'UC8Oy99fOvCjHfbvTImDXpkg', 'UCgvlSYolCHq5JiiosIvM6lw', 'UCcxxhVKa9wVvHXscwMXC2gQ', 'UCs5NZ-lIbR_rvYDaKvBrVFw',
'UCHCtAgNRSrcBMRqU5SfHplw', 'UC07gLBHCOrcOyE0bw8Lv8Jg', 'UCnOW6JnwbE46hKOPyDYxc_w', 'UCLlW4UJkWDqW9Ht4u8LRztw',
'UCoPBjpeflrKudIDsvDEV0aw', 'UCcoKvgQgWLXosFP5giUeI4g', 'UChYGnlCGDagZT0F8GYhDY2w', 'UCapA77PmpW9dEZQxxJY7TIg',
'UCfdoVOJ9krvIZReKoLWAelA', 'UCU5WWthBfCSYPt-YRNRZYPQ', 'UCDDbjiG_3PHqn4_8BRMr9yA', 'UCVnjpK8HoaeYGQCJUjtiq6g',
'UCkJePyY8lKMvWSF9S-X_hpQ', 'UCcoKvgQgWLXosFP5giUeI4g', 'UCcoKvgQgWLXosFP5giUeI4g', 'UCwRfjTiJGrONLlNHFQ0mQ2A',
'UCJoDuLaZaOmmTLk9IVst6OA', 'UCk4bBXoqPm8j2FjyStas7jQ', 'UC6M_UOk3D8ZG0J2HmuK6J7g', 'UCkJePyY8lKMvWSF9S-X_hpQ',
'UCk4bBXoqPm8j2FjyStas7jQ', 'UCgHM7Mjvat503rAX9jlFf3w', 'UCzywEh2M-ReJTvlPh-jA1-w', 'UC1Wi-CaZ_u111EET3es-jzA',
'UClmn1s3aSJcqsGoQG7ZV5rg', 'UCIAYvaVP15Cel4NZ1ib_ADA', 'UC8Oy99fOvCjHfbvTImDXpkg', 'UCcxxhVKa9wVvHXscwMXC2gQ', 
'UCHCtAgNRSrcBMRqU5SfHplw', 'UC07gLBHCOrcOyE0bw8Lv8Jg', 'UCHCtAgNRSrcBMRqU5SfHplw', 'UCk4bBXoqPm8j2FjyStas7jQ', 
'UClmXrUcfU0A3BWSCU1XouNg', 'UCDhBrg4uKxdc0u5mv2x9GAA', 'UCwgMHGG8IDcYjjSXtYbE9rQ', 'UChYGnlCGDagZT0F8GYhDY2w', 
'UCcoKvgQgWLXosFP5giUeI4g', 'UC4A5Ov8NNZ8K1Lq5cfcESUg', 'UC7qpkq26VEDSe6H3ZwTMWJg', 'UCgvlSYolCHq5JiiosIvM6lw',
'UCk44Jx55IZkD57On49NFtiw', 'UCPDuYbsUJxNexaCBruG-jkw', 'UCOqxwnL0H8oCTHlBNO9uSGQ', 'UCnOW6JnwbE46hKOPyDYxc_w', 
'UCblmGPQJ1_O_gBMmGRiVM9Q', 'UC4IerKEP2BDFd2VVXLaeU7w', 'UCVnjpK8HoaeYGQCJUjtiq6g', 'UChoTld7IoAinX5E7nYTZJVg',
'UCwgMHGG8IDcYjjSXtYbE9rQ', 'UC4IerKEP2BDFd2VVXLaeU7w', 'UCYbUqEvxVhVh71wBbrgvyGA', 'UCEmzN-PVaElRDmdlBFbMW1A', 
'UCNBOZg8v0cFe2RYvSgJ_K3A', 'UCpTKksadLBwHiFOdtxVq5Sg', 'UCkGvXvq42A152M8S5xOMbBw', 'UC4kPR2QLYHmURMMr5ERPAcA', 
'UCJ4a59zmaTKVThYBU-ZZdFA', 'UCV_Qh1S9gCCXm2XBQRy0v8Q', 'UCsXIMkerN0ofYQVIvNWr7Dg', 'UCgP6tTK8LrlhK7LMhExs6eQ', 
'UCRqHZqWr81nXW0Qr2MVcD-g', 'UCpTKksadLBwHiFOdtxVq5Sg', 'UC1Wi-CaZ_u111EET3es-jzA', 'UCfH5_gBNYMszkN-6P1SAC1Q', 
'UCfH5_gBNYMszkN-6P1SAC1Q', 'UC8Oy99fOvCjHfbvTImDXpkg', 'UCYGbbdOfUAtBOFoiifA4cpw', 'UCFSHMdKANs7Ee_xAfvfX7ug', 
'UChYuxekYJM1HXY6ThthPR8A', 'UCXLbuE7gWN6CVyliQHKP62w', 'UC1Wi-CaZ_u111EET3es-jzA', 'UCYGbbdOfUAtBOFoiifA4cpw',
'UCpd3kplR9EdsvjbCAtp1JPw', 'UCOLzBAcpeCiue5PnlPSvczQ', 'UCpd3kplR9EdsvjbCAtp1JPw', 'UCMhUkjPmAi9xrbskMrdNPWA',
'UCLlW4UJkWDqW9Ht4u8LRztw', 'UC0anupzl59WSN3XiwtcskaA', 'UCOLzBAcpeCiue5PnlPSvczQ', 'UClLJBDGFBNEvfuqO7mnISDA',
'UCOLzBAcpeCiue5PnlPSvczQ', 'UCnOW6JnwbE46hKOPyDYxc_w', 'UCINx9NBUBTrtejzu6j_Msrw', 'UCnOW6JnwbE46hKOPyDYxc_w',
'UCINx9NBUBTrtejzu6j_Msrw'
]

def isScam(id):

    if  id in scamChannels:
        return 1
    else:
        return 0


data['Is Scam'] = data['AuthorID'].apply(lambda x: isScam(x))

train = data[0:30000].copy()[['Comments', 'Is Reply', 'Author', 'Is Scam']]
test = data[30001:35001].copy()[['Comments', 'Is Reply', 'Author', 'AuthorID']]

train['Word Count'] = train['Comments'].apply(lambda x: len(str(x).split()))

#convert to lowercase, strip and remove punctuations
def preprocess(text):
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
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
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
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))


train['Clean Comment'] = train['Comments'].apply(lambda x: finalpreprocess(x))
train['Clean Username'] = train['Author'].apply(lambda x : finalpreprocess(x))

train['Clean text tokens']=[nltk.word_tokenize(i) for i in train['Clean Comment']]
train['Clean Username tokens'] = [nltk.word_tokenize(i) for i in train['Clean Username']]

X_train, X_test, y_train, y_test = train_test_split(train['Clean Comment'], train['Is Scam'], test_size=0.4, shuffle=True)
X_train_author, X_test_author, y_train_author, y_test_author = train_test_split(train['Clean Username'], train['Is Scam'], test_size=0.4, shuffle=True)

#Tf-Idf
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)

X_train_author_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train_author)
X_test_author_vectors_tfidf = tfidf_vectorizer.transform(X_test_author)

#FITTING THE CLASSIFICATION MODEL using Logistic Regression(tf-idf)
lr_tfidf=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
lr_tfidf.fit(X_train_vectors_tfidf, y_train)  #model

y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]

print(classification_report(y_test,y_predict))
print('Confusion Matrix:',confusion_matrix(y_test, y_predict))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

#Pre-processing the new dataset
test['Clean Comments'] = test['Comments'].apply(lambda x: finalpreprocess(x)) #preprocess the data

X_test = test['Clean Comments']
#converting words to numerical data using tf-idf
X_vector=tfidf_vectorizer.transform(X_test)
###X_vector_author = tfidf_vectorizer.transform(X_test_author)
#use the best model to predict 'target' value for the new dataset 
y_predict = lr_tfidf.predict(X_vector)      
y_prob = lr_tfidf.predict_proba(X_vector)[:,1]

###y_predict_author = lr_tfidf.predict(X_vector_author)      
###y_prob_author = lr_tfidf.predict_proba(X_vector_author)[:,1]

test['predict_prob']= y_prob
test['Is Scam']= y_predict
###test['author_predict_prob'] = y_prob_author
###test['Is Scam _author'] = y_predict_author
final=test[['Clean Comments', 'Clean Username', 'predict_prob', 'Is Scam', 'AuthorID']].reset_index(drop=True)