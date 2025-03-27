#!/usr/bin/env python
# coding: utf-8

# # Uber review dataset

# #Python Libraries for scraping reviews

# In[3]:


pip install google_play_scraper


# In[10]:


from google_play_scraper import app, reviews_all


# In[6]:


import pandas as pd
import numpy as np


# #Code for scraping data

# In[13]:


uber_review = reviews_all("com.ubercab", sleep_milliseconds=0, lang="en", country="NG")
#print(uber_review)
df = pd.json_normalize(uber_review)
print(df.head())


# In[ ]:


df.to_csv("uber_review.csv")


# In[7]:


df.shape


# In[8]:


df.info()


# # Text processing and Analysis

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re


# #Categorizing the data into 3 namely Interface, Navigation and Responsiveness

# In[5]:


df = pd.read_csv('uber_review.csv')


# In[20]:


# Define categories based on keywords
categories = {
    'user_interface_keywords': ['interface', 'design', 'layout'],
    'responsiveness_keywords': ['responsive', 'speed', 'fast'],
    'navigation_keywords': ['navigation', 'menu', 'usability']
}


# In[21]:


# Assign categories to each review
for category, keywords in categories.items():
    df[category] = df['content'].apply(lambda x: any(keyword in x for keyword in keywords))


# In[22]:


# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['content'], df[['user_interface_keywords', 'responsiveness_keywords', 'navigation_keywords']],
    test_size=0.2, random_state=42
)


# In[23]:


# Create a Bag-of-Words model
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)


# In[24]:


# Train a Multinomial Naive Bayes classifier for each category
classifiers = {}
for category in ['user_interface_keywords', 'responsiveness_keywords', 'navigation_keywords']:
    classifier = MultinomialNB()
    classifier.fit(train_features, train_labels[category])
    classifiers[category] = classifier


# In[25]:


# Make predictions on the test set for each category
predictions = pd.DataFrame()
for category, classifier in classifiers.items():
    predictions[category] = classifier.predict(test_features)


# In[26]:


# Evaluate the model
for category in classifiers.keys():
    accuracy = accuracy_score(test_labels[category], predictions[category])
    classification_rep = classification_report(test_labels[category], predictions[category])
    
    print(f"Category: {category}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", classification_rep)
    print("------------------------")


# In[4]:


#SVM Evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create a new 'Category' column based on existing keyword columns
df['Category'] = df[['user_interface_keywords', 'responsiveness_keywords', 'navigation_keywords']].idxmax(axis=1)
    
# Split the dataset into features (X) and labels (y)
X = df['content']
y = df['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', labels=y.unique())
recall = recall_score(y_test, y_pred, average='weighted', labels=y.unique())
f1 = f1_score(y_test, y_pred, average='weighted', labels=y.unique())
conf_matrix = confusion_matrix(y_test, y_pred, labels=y.unique())

# Print the evaluation metrics
print("\nEvaluation for SVM \n")
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Measure: {f1:.2f}')

# Print the confusion matrix
print('\nConfusion Matrix:\n')
print(conf_matrix)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Split the dataset into features (X) and labels (y)
X = df['content']
y = df['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = decision_tree_classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', labels=y.unique())
recall = recall_score(y_test, y_pred, average='weighted', labels=y.unique())
f1 = f1_score(y_test, y_pred, average='weighted', labels=y.unique())
conf_matrix = confusion_matrix(y_test, y_pred, labels=y.unique())

# Print the evaluation metrics for Decision Tree
print("\nDecision Tree Classifier Evaluation:\n")
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Print the confusion matrix for Decision Tree
print('\nConfusion Matrix:\n')
print(conf_matrix)

# # Analysing Interface Reviews

# In[27]:


interface = pd.read_csv('interface_reviews.csv')
interface


# In[127]:


Data = (interface['content'].str.lower())
Data.head()


# In[128]:


def remove_punctuation(Data):
    Data_nopunct = "".join([c for c in Data if c not in string.punctuation])
    return Data_nopunct
cleaned_data = Data.apply(lambda x: remove_punctuation(x))
cleaned_data.head()


# In[129]:


def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens
Tokenized_data = cleaned_data.apply(lambda x: tokenize(x))
Tokenized_data.head()


# In[130]:


stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(Tokenized_data):
    no_stopwords = [word for word in Tokenized_data if word not in stopwords]
    return no_stopwords
New_dataset = Tokenized_data.apply(lambda x: remove_stopwords(x))
New_dataset.head()


# In[131]:


stemmer = PorterStemmer()
def stemming(New_content):
    new_words = [stemmer.stem(word) for word in New_content]
    return ' '.join(New_content)
stemmed_data = New_dataset.apply(lambda x: stemming(x))
stemmed_data.head()


# In[134]:


my_list = stemmed_data.tolist()
my_list


# In[34]:


word_list = [word for sentence in my_list for word in sentence.split()]
word_list


# #Compare a list of positive words with the word_list

# In[35]:


positive_words = [
    'amazing', 'awesome', 'brilliant', 'excellent', 'fantastic', 'good',
    'great', 'outstanding', 'superb', 'terrific',
    'wonderful', 'perfect', 'extraordinary', 'superior', 'best',
    'top-notch', 'splendid', 'impressive', 'joyful', 'happy',
    'satisfying', 'delightful', 'glorious', 'fabulous', 'magnificent', 'beautiful'
]


# In[36]:


#Append the words that match into a positive emotion list
positive_emotion = []
for word in word_list:
    if word in positive_words:
        positive_emotion.append(word)
positive_emotion


# In[37]:


#Check the number of words that match
len(positive_emotion)


# #Use the counter function to get the frequency of each match

# In[38]:


from collections import Counter


# In[40]:


w = Counter(positive_emotion)
w


# # Bar-chart for the interface positive emotion

# #Install the matplotlib to plot the graph

# In[41]:


pip install matplotlib


# In[42]:


import matplotlib.pyplot as plt


# In[43]:


bar_width = 0.5
plt.figure(figsize=(14, 6))
plt.bar(w.keys(), w.values(), width=bar_width)
plt.title('Interface_positive_emotion')
plt.xlabel('Emotions')
plt.ylabel('Frequency')
plt.show()


# # Word cloud for interface positive emotion

# In[44]:


pip install wordcloud


# In[46]:


from wordcloud import WordCloud


# In[48]:


#Convert list to string
positive_all_emotion = ', '.join(t for t in positive_emotion)
positive_all_emotion


# In[49]:


# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_all_emotion)


# In[50]:


# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis labels
plt.title('Interface Positive Emotion')
plt.show()


# #Compare a list of negative words with the word_list with For Loop

# In[51]:


negative_words = [
    'bad', 'awful', 'terrible', 'horrible', 'negative', 'poor',
    'unpleasant', 'dislike', 'disappointing', 'inferior', 'dreadful',
    'lousy', 'unfavorable', 'disgusting', 'hateful', 'abysmal', 'mediocre',
    'atrocious', 'annoying', 'miserable', 'unfortunate', 'worst', 'grim',
    'unhappy', 'unappealing', 'repugnant', 'distasteful', 'abhorrent',
    'grim', 'appalling', 'insufferable', 'detestable', 'ghastly', 'repulsive', 'unsatisfied', 
]


# In[52]:


negative_emotion = []
for word in word_list:
    if word in negative_words:
        negative_emotion.append(word)
negative_emotion


# #Check the number of negative words

# In[53]:


len(negative_emotion)


# #Count the number of times each word appears

# In[54]:


N = Counter(negative_emotion)
N


# # Bar chart for negative emotion

# In[ ]:


bar_width = 0.5
plt.figure(figsize=(20, 6))
plt.bar(N.keys(), N.values(), width=bar_width)
plt.title('Interface_negative_emotion')
plt.xlabel('Emotions')
plt.ylabel('Frequency')
plt.show()


# # Word cloud interface negative emotion

# In[56]:


negative_all_emotion = ', '.join(t for t in negative_emotion)
negative_all_emotion


# In[57]:


# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_all_emotion)


# In[58]:


# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis labels
plt.title('Interface Negative Emotion')
plt.show()


# # Sentiments bar chart

# In[59]:


sentiments = ['Positive', 'Negative']
frequencies = [217, 277] 
plt.bar(sentiments, frequencies, color=['green', 'red'])
plt.xlabel('Sentiments')
plt.ylabel('Frequency')
plt.title('Comparing Positive and Negative Sentiments')
for i, freq in enumerate(frequencies):
    plt.text(i, freq + 5, str(freq), ha='center', va='bottom')
plt.show()


# # Analysing Navigation Reviews

# In[135]:


navigation = pd.read_csv('navigation_reviews.csv')
navigation.head()


# In[136]:


Data2 = (navigation['content'].str.lower())
Data2.head()


# In[137]:


def remove_punctuation(Data2):
    Data_nopunct = "".join([c for c in Data2 if c not in string.punctuation])
    return Data_nopunct
cleaned_data2 = Data2.apply(lambda x: remove_punctuation(x))
cleaned_data2.head()


# In[138]:


def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens
Tokenized_data2 = cleaned_data2.apply(lambda x: tokenize(x))
Tokenized_data2.head()


# In[139]:


stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(Tokenized_data2):
    no_stopwords = [word for word in Tokenized_data2 if word not in stopwords]
    return no_stopwords
New_dataset2 = Tokenized_data2.apply(lambda x: remove_stopwords(x))
New_dataset2.head()


# In[140]:


stemmer = PorterStemmer()
def stemming(New_content):
    new_words = [stemmer.stem(word) for word in New_content]
    return ' '.join(New_content)
stemmed_data2 = New_dataset2.apply(lambda x: stemming(x))
stemmed_data2.head()


# In[66]:


my_list2 = stemmed_data2.tolist()
my_list2


# In[67]:


word_list2 = [word for sentence in my_list2 for word in sentence.split()]
word_list2


# In[68]:


positive_navigation_words = [
    'smooth', 'effortless', 'easy', 'seamless', 'intuitive', 'user-friendly',
    'convenient', 'straightforward', 'accessible', 'simple', 'hassle-free',
    'clear', 'efficient', 'well-organized', 'navigable', 'responsive', 'streamlined',
    'pleasurable', 'user-centric', 'simplified', 'straightforward', 'logical',
    'well-designed', 'swift', 'uncomplicated', 'user-focused', 'fluid', 'clear-cut'
]


# In[70]:


positive_navigation_emotion = []
for word in word_list2:
    if word in positive_navigation_words:
        positive_navigation_emotion.append(word)
positive_navigation_emotion


# #Check the number of words and count the number of times each word appears

# In[71]:


len(positive_navigation_emotion)


# In[72]:


pw = Counter(positive_navigation_emotion)
pw


# # Bar chart for positive navigation emotion

# In[ ]:


bar_width = 0.5
plt.figure(figsize=(14, 6))
plt.bar(pw.keys(), pw.values(), width=bar_width)
plt.title('Navigation_positive_emotion')
plt.xlabel('Emotions')
plt.ylabel('Frequency')
plt.show()


# In[74]:


positive_all_emotion2 = ', '.join(t for t in positive_navigation_emotion)
positive_all_emotion2


# # Word cloud for navigation positive emotion

# In[95]:


# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_all_emotion2)


# In[96]:


# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis labels
plt.title('Navigation Positive Emotion')
plt.show()


# In[77]:


negative_navigation_words = [
    'confusing', 'complex', 'complicated', 'difficult', 'challenging', 'frustrating',
    'cumbersome', 'clumsy', 'convoluted', 'counterintuitive', 'disorganized',
    'disorienting', 'unintuitive', 'awkward', 'intricate', 'tedious', 'bewildering',
    'jumbled', 'muddled', 'obtuse', 'overwhelming', 'unwieldy', 'unmanageable',
    'chaotic', 'hard-to-navigate', 'problematic', 'annoying', 'not user-friendly'
]


# In[79]:


negative_navigation_emotion = []
for word in word_list2:
    if word in negative_navigation_words:
        negative_navigation_emotion.append(word)
negative_navigation_emotion


# In[85]:


len(negative_navigation_emotion)


# In[86]:


nw = Counter(negative_navigation_emotion)
nw


# # Bar chart for navigation negative emotion

# In[88]:


bar_width = 0.5
plt.figure(figsize=(18, 6))
plt.bar(nw.keys(), nw.values(), width=bar_width)
plt.title('Navigation_negative_emotion')
plt.xlabel('Emotions')
plt.ylabel('Frequency')
plt.show()


# In[89]:


negative_all_emotion2 = ', '.join(t for t in negative_navigation_emotion)
negative_all_emotion2


# # Word cloud for navigation negative emotion

# In[90]:


# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_all_emotion2)


# In[92]:


# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis labels
plt.title('Navigation negative Emotion')
plt.show()


# # Sentiment Analysis for Navigation

# In[93]:


sentiments = ['Positive', 'Negative']
frequencies = [189, 213] 
plt.bar(sentiments, frequencies, color=['green', 'red'])
plt.xlabel('Sentiments')
plt.ylabel('Frequency')
plt.title('Comparing Positive and Negative Sentiments for Navigation')
for i, freq in enumerate(frequencies):
    plt.text(i, freq + 5, str(freq), ha='center', va='bottom')
plt.show()


# # Analysis for Responsiveness

# In[141]:


responsiveness = pd.read_csv('responsiveness_reviews.csv')
responsiveness.head()


# In[142]:


Data3 = (responsiveness['content'].str.lower())
Data3.head()


# In[143]:


def remove_punctuation(Data3):
    Data_nopunct = "".join([c for c in Data3 if c not in string.punctuation])
    return Data_nopunct
cleaned_data3 = Data3.apply(lambda x: remove_punctuation(x))
cleaned_data3.head()


# In[144]:


def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens
Tokenized_data3 = cleaned_data3.apply(lambda x: tokenize(x))
Tokenized_data3.head()


# In[145]:


stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(Tokenized_data3):
    no_stopwords = [word for word in Tokenized_data3 if word not in stopwords]
    return no_stopwords
New_dataset3 = Tokenized_data3.apply(lambda x: remove_stopwords(x))
New_dataset3.head()


# In[146]:


stemmer = PorterStemmer()
def stemming(New_content):
    new_words = [stemmer.stem(word) for word in New_content]
    return ' '.join(New_content)
stemmed_data3 = New_dataset3.apply(lambda x: stemming(x))
stemmed_data3.head()


# In[106]:


my_list3 = stemmed_data3.tolist()
my_list3


# In[107]:


word_list3 = [word for sentence in my_list3 for word in sentence.split()]
word_list3


# #Compare word_list3 with the list of positive_responsiveness_words

# In[108]:


positive_responsiveness_words = [
    'quick', 'fast', 'responsive', 'speedy', 'efficient', 'prompt', 'instant',
    'swift', 'rapid', 'immediate', 'snappy', 'timely', 'smooth', 'seamless',
    'instantaneous', 'speedy', 'agile', 'nimble', 'lightning-fast', 'energetic',
    'brisk', 'lively', 'vigorous', 'brisk', 'nimble', 'prompt', 'fleet', 'active',
    'quick-witted', 'accelerated', 'dynamic', 'ready', 'expeditious'
]


# In[109]:


positive_responsiveness_emotion = []
for word in word_list3:
    if word in positive_responsiveness_words:
        positive_responsiveness_emotion.append(word)
positive_responsiveness_emotion


# In[111]:


len(positive_responsiveness_emotion)


# In[112]:


p_w = Counter(positive_responsiveness_emotion)
p_w


# # Bar chart for responsiveness positive emotion

# In[113]:


bar_width = 0.5
plt.figure(figsize=(14, 6))
plt.bar(p_w.keys(), p_w.values(), width=bar_width)
plt.title('Responsiveness_positive_emotion')
plt.xlabel('Emotions')
plt.ylabel('Frequency')
plt.show()


# # Word cloud for responsiveness positive emotion

# In[120]:


positive_all_emotion3 = ', '.join(t for t in positive_responsiveness_emotion)
positive_all_emotion3


# In[123]:


# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_all_emotion3)


# In[122]:


# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis labels
plt.title('Responsiveness Positive Emotion')
plt.show()


# In[114]:


negative_responsiveness_words = [
    'slow', 'sluggish', 'unresponsive', 'laggy', 'delayed', 'lethargic',
    'tardy', 'unprompt', 'tardy', 'inactive', 'torpid', 'indolent', 'dull',
    'slack', 'leisurely', 'sluggardly', 'lagging', 'slumberous', 'lagging',
    'ponderous', 'unhurried', 'listless', 'unwieldy', 'delinquent', 'slumberous',
    'languid', 'dawdling', 'lagging', 'idle', 'delayed', 'unquick', 'inactive'
]


# In[115]:


negative_responsiveness_emotion = []
for word in word_list3:
    if word in negative_responsiveness_words:
        negative_responsiveness_emotion.append(word)
negative_responsiveness_emotion


# In[116]:


len(negative_responsiveness_emotion)


# In[117]:


n_w = Counter(negative_responsiveness_emotion)
n_w


# # Bar chart for responsiveness negative emotion

# In[118]:


bar_width = 0.5
plt.figure(figsize=(18, 6))
plt.bar(n_w.keys(), n_w.values(), width=bar_width)
plt.title('Responsiveness_negative_emotion')
plt.xlabel('Emotions')
plt.ylabel('Frequency')
plt.show()


# In[124]:


negative_all_emotion3 = ', '.join(t for t in negative_responsiveness_emotion)
negative_all_emotion3


# # Word cloud for responsiveness negative emotion

# In[125]:


# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_all_emotion3)


# In[126]:


# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis labels
plt.title('Responsiveness Negative Emotion')
plt.show()


# # Sentiment Analysis for Responsiveness

# In[119]:


sentiments = ['Positive', 'Negative']
frequencies = [46, 11] 
plt.bar(sentiments, frequencies, color=['green', 'red'])
plt.xlabel('Sentiments')
plt.ylabel('Frequency')
plt.title('Comparing Positive and Negative Sentiments for Responsiveness')
for i, freq in enumerate(frequencies):
    plt.text(i, freq + 5, str(freq), ha='center', va='bottom')
plt.show()

