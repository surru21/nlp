from flask import Flask, request, redirect, url_for, render_template
import pandas as pd
import numpy as np
#from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
from string import punctuation

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
tfidf=pickle.load(open('transform.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
        return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
       if request.method == 'POST':
         message = request.form['message']
         data = [message]
         all_stopwords_gensim = STOPWORDS
         all_punct = set(punctuation)
         all_stopwords_gensim1 = STOPWORDS.union(set(all_punct))                                                           
         all_stopwords_gensim2 = all_stopwords_gensim1.union(set(['hello','cc','sir', 'hi', 'hallo','lady','gentleman','man','hey', 'received from', 'please']))                    
         sw_list = {"cant","can't","cannot","not","@"}                                                 
         all_stopwords_gensim = all_stopwords_gensim2.difference(sw_list)                  
         data=data.lower()                              
         data=data.replace("_"," ")          
         data=data.replace("-"," ")          
         data=data.replace("#"," ")          
         data=data.replace("/"," ")          
         data=data.replace(":"," ")  
         data1 = word_tokenize(data) 
         data2 = [word for word in data1 if not word in all_stopwords_gensim]
         data2 = str(data2)
         data3 = data2.replace(r'\d', '')              
         data3 = data3.replace(r'\S*@\S*\s?', '')      
         data3 = data3.replace(r'[^\x00-\x7f]+', '')                             
         data3 = data3.replace(",","")
         data3 = data3.replace("'","")
         data3 = data3.replace("[","")
         data3 = data3.replace("]","")
         vect = tfidf.transform(data3).toarray()
         my_prediction = clf.predict(vect)
       return render_template('result.html',prediction = my_prediction) 

if __name__ == '__main__':
     app.run(debug=True, use_reloader=False)  
