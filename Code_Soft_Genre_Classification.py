# importing required packages
import pandas as pd
import numpy as np
data= pd.read_csv("train_data.txt",sep=":::",names=["S_No","movie_name","genre","summary"],engine='python')
data.to_csv("train_data.csv",index=None)
data=data.drop("S_No",axis=1)
# Removing the years from movie names as they are not usefull
for i in range(0,len(data["movie_name"])):
    temp=data["movie_name"][i].split(" ")
    data["movie_name"][i]=" ".join(temp[0:len(temp)-2])
#Data Exploration
data.head()
data.isnull().sum()
data.shape
data.head()
data.tail()
# Splitting into feature and target datasets
X = data.drop("genre",axis=1)
Y = data["genre"]
# Using natural language processing tool kit and regular expression modules to extract usefull insights from the summary

import nltk
import re
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
l = WordNetLemmatizer()
corpus=[]
all_stop_words=stopwords.words("english")
all_stop_words.remove("not")
all_stop_words.remove("didn't")
# Lemmatizing each word in the summary and removing all unuseful stopwords
for i in range(0,len(X["summary"])):
     summary = re.sub("[^a-zA-Z]",' ',X["summary"][i])
     summary = summary.lower()
     summary = summary.split()
     summary = [l.lemmatize(word) for word in summary if  word not in all_stop_words]
     summary = " ".join(summary)
     X["summary"][i]=summary
X.head()
X.tail()
X.shape
Y.shape
X.head()
X.tail()
Y.head()
Y.tail()
# Feature extraction using TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=54214,)
# X["summary"]=tfidf.fit_transform(X["summary"]).toarray()
a =tfidf.fit_transform(X["summary"])
print(a)
#Splitting into training and testing sets and using test dataset size as 20%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(a,Y,test_size=0.20,shuffle=True,stratify=Y)
#appliying Multinomial Naive Bayes for best results
from sklearn.naive_bayes import MultinomialNB
nb= MultinomialNB()
#training model
nb.fit(X_train,Y_train)
nb.score(X_train,Y_train)
#testing
pred = nb.predict(X_test)
# Reporting results
from sklearn.metrics import confusion_matrix,classification_report
matrix = confusion_matrix(Y_test,pred)
report = classification_report(Y_test,pred)
print(matrix)
print(report)

    
