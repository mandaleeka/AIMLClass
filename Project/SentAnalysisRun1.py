
import configparser
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib import style
style.use('ggplot')

import nltk
nltk.download('stopwords')



from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from collections import Counter


from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer

from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word

from html.parser import HTMLParser

import os
import warnings
warnings.filterwarnings('ignore')


###### Import all configuraion information
config = configparser.ConfigParser()
config.read('config.ini')
#       Data source
dataSourceType = config.get('DataSource', 'type')
dataSourceFile = config.get('DataSource', 'file')
#       language
language = config.get('Language', 'lang')
tokenToBeRemoved = config.get('Token', 'tokenToBeRemoved')
#       input type
useCaseType = config.get('UseCase', 'useCase')

included_tags = {"NOUN", "ADJ", "ADV"}

date_regex = r'[0-9]{1,2}[\/,:][0-9]{1,2}[\/,:][0-9]{2,4}'
email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_regex = r'((?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}))'
special_char_regex = r'[^a-zA-z0-9\s]'
########

#### get Data
input_data = pd.read_csv(dataSourceFile)
#print(input_data.head(10))

#print("\n\n", useCaseType, "\n\n")



#importing the data
print(input_data.shape)
#Summary of the dataset
print(input_data.describe())
#sentiment count
print(input_data['sentiment'].value_counts())


######  Data Cleanup ###############################
#Setting stopwords
stop_words = set(stopwords.words(language))
#print(stop_words)
print(tokenToBeRemoved)


#Removing the html strips
class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []

    def handle_data(self, data):
        self.result.append(data)

def clean_html(html):
    parser = MyHTMLParser()
    parser.feed(html)
    return ''.join(parser.result)


#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def sentence_tokenizer(text=None):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(date_regex, ' ', text)
    text = re.sub(email_regex, ' ', text)
    text = re.sub(phone_regex, ' ', text)
    text = re.sub(r"\s+", ' ', text)
    text = clean_html(text)
    text = remove_between_square_brackets(text)
    # print("Text after tokenizing " + text)
    doc = nlp(text)
    return [token.lemma_ for token in doc
            if not token.is_stop
            and token.text not in tokenToBeRemoved
            and not token.is_punct
            and not token.text.isspace()
            and not token.text.isdigit()
            and token.pos_ in included_tags
            and token.text.isalpha()
            and len(token.text.strip()) > 3]

#Removing the unnecessary text
def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub('<br />', '', text)
    text = re.sub(date_regex, ' ', text)
    text = re.sub(email_regex, ' ', text)
    text = re.sub(phone_regex, ' ', text)
    text = re.sub(special_char_regex, ' ', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r"\s+", ' ', text)

    text = clean_html(text)
    text = remove_between_square_brackets(text)
    return text

#Apply function on review column
input_data['review']=input_data['review'].apply(clean_text)
#print(input_data.head(10))

# Remove stop words from data
def remove_stopwords(text):
    #tokens = tokenizer.tokenize(text)
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if ((token not in stop_words) and (token not in tokenToBeRemoved))]
    #filtered_tokens = [token for token in tokens if token not in tokenToBeRemoved]

    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

input_data['review']=input_data['review'].apply(remove_stopwords)

print(input_data.head(1))
'''
#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
input_data['review']=input_data['review'].apply(simple_stemmer)
'''

#print(input_data.head(10))
#####################################################################

## EDA

# Count the number of instances of each sentiment category
sentiment_counts = input_data['sentiment'].value_counts()
# Create a bar chart of the sentiment categories
plt.bar(sentiment_counts.index, sentiment_counts.values)
# Add chart title and axis labels
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Samples')
# Display the chart
#plt.show()

def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count

input_data['word count'] = input_data['review'].apply(no_of_words)

fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].hist(input_data[input_data['sentiment'] == 'positive']['word count'], label = 'Positive', color='blue', rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(input_data[input_data['sentiment'] == 'negative']['word count'], label = 'Negative', color='red', rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle("Number of words in review")
#plt.show()

#####################################################################

## Remove duplicate data
duplicates = input_data.duplicated().sum()
input_data = input_data.drop_duplicates()
#print("\n\n", "Duplicates ", duplicates, "\n\n")
#print(input_data.head(10))

#####################################################################
#  Find most commonly used words 

pos_reviews =  input_data[input_data.sentiment == 'positive']
#print("Positive reviews", "\n\n", pos_reviews.head())

count = Counter()
for text in pos_reviews['review'].values:
    for word in text.split():
        count[word] +=1
count.most_common(15)

pos_words = pd.DataFrame(count.most_common(15))
pos_words.columns = ['word', 'count']
print(pos_words.head(15))

#plt.bar(pos_words, x='count', y='word', title='Common words in positive reviews', color = 'word')


neg_reviews =  input_data[input_data.sentiment == 'negative']

count_neg = Counter()   
for text in neg_reviews['review'].values:
    for word in text.split():
        count_neg[word] +=1
count_neg.most_common(15)

neg_words = pd.DataFrame(count_neg.most_common(15))
neg_words.columns = ['word', 'count']
print(neg_words.head(15))

#plt.bar( neg_words, x='count', y='word', title='Common words in negative reviews', color = 'word')



###1111    Spliting data to train and to test
#split the dataset  

Reviews = input_data['review']
Sentiments = input_data['sentiment']

vect = TfidfVectorizer()
Reviews = vect.fit_transform(input_data['review'])

x_train, x_test, y_train, y_test = train_test_split(Reviews, Sentiments, test_size=0.2, random_state=42)


print("Size of x_train: ", (x_train.shape))
print("Size of y_train: ", (y_train.shape))
print("Size of x_test: ", (x_test.shape))
print("Size of y_test: ", (y_test.shape))


########## Logistic Regression training
print("##########               Logistic Regression          ########## ")
logReg = LogisticRegression()
logReg.fit(x_train, y_train)
logReg_pred = logReg.predict(x_test)
logReg_acc = accuracy_score(logReg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logReg_acc*100))

print(confusion_matrix(y_test, logReg_pred))
print("\n")
print(classification_report(y_test, logReg_pred))
print("\n")

###############################################################

########## Multinomial Naive Bayes model
print("##########               Multinomial Naive Bayes model          ########## ")
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
mnb_pred = mnb.predict(x_test)
mnb_acc = accuracy_score(mnb_pred, y_test)
print("Test accuracy: {:.2f}%".format(mnb_acc*100))

print(confusion_matrix(y_test, mnb_pred))
print("\n")
print(classification_report(y_test, mnb_pred))
print("\n")

###############################################################

########## Support Vector Classifier model
print("##########               Support Vector Classifier model         ########## ")
svc = LinearSVC()
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("Test accuracy: {:.2f}%".format(svc_acc*100))

print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred))
print("\n")

###############################################################
