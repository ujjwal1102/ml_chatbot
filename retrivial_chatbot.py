import string
import nltk 

import numpy as np
import random
import string

import bs4 as bs
import re
import requests
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



warnings.filterwarnings = False


nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')

r = requests.get('https://en.wikipedia.org/wiki/Cuisine')
raw_html = r.text
# print(r)
########################################### Cleaning up

corpus_html = bs.BeautifulSoup(raw_html)
# print(corpus_html)
############################################ extraction paragraphs from the html

corpus_paras = corpus_html.find_all('p')
corpus_text = ' '
# print(corpus_paras)

########################################### Concatenation all the paras
for para in corpus_paras:
    corpus_text += para.text
        
# print(corpus_text)
corpus_text = corpus_text.lower()
# print(corpus_text)

######################################## cleaning data [ Empty spaces and special characters]

corpus_text = re.sub(r'[[0-9]*\]',' ',corpus_text)
corpus_text = re.sub(r'\s+',' ',corpus_text)
# print(corpus_text)
######################################## converting the text into sentences and word tokens


corpus_sentences = nltk.sent_tokenize(corpus_text)
corpus_words = nltk.word_tokenize(corpus_text)

# print(corpus_words)

greeting_inputs = ['hey','hi','good morning','good evening','what\'s up']
greeting_responses = ['hello','how are you','*nods*','hello dear, how are you','Welcome']
length = len(greeting_responses)
def greet_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)
            
            
############### processing with punchuatinon removaland lemmitizing

wn_lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_corpus(tokens):
    return [wn_lemmatizer.lemmatize(token) for token in tokens]

punct_removal_dict = dict((ord(punctuation),None) for punctuation in string.punctuation)
# print(punct_removal_dict)

def get_processed_text(document):
    return lemmatize_corpus(nltk.word_tokenize(document.lower().translate(punct_removal_dict)))
        

############### Language Modeling with tf-idf

def respond(user_input):
    bot_response = ' '
    corpus_sentences.append(user_input)

    ##vectorizing the processed text

    word_vectorizer = TfidfVectorizer(tokenizer= get_processed_text, stop_words = 'english')
    corpus_word_vectors = word_vectorizer.fit_transform(corpus_sentences)

    cos_sin_vectors = cosine_similarity(corpus_word_vectors[-1], corpus_word_vectors)
    similar_response_idx = cos_sin_vectors.argsort()[0][-2]

    matched_vector = cos_sin_vectors.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        bot_response = bot_response + "i\'m sorry, what is it again?"
        return bot_response

    else: 
        bot_response = bot_response + corpus_sentences[similar_response_idx]

        return bot_response

######################################################


chat = True
print("Hello what do you want to learn about cuisine today?")
while( chat == True):
    user_query = input()
    user_query = user_query.lower()
    if user_query != "quit":
        if user_query == 'thanks' or user_query == 'thankyou':
            chat = False
            print('CuisineBot: Your Welcome!')

        else:
            if greet_response(user_query) != None:
                print("CuisineBot: " + greet_response(user_query))
            print("CuisineBot: ",end = "")
            print(respond(user_query))
            corpus_sentences.remove(user_query)
    else:
        chat = False
        print("CuisineBot: Good bye!")


