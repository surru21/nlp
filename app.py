from flask import Flask, request, redirect, url_for, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle

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
         data=data.replace("_"," ")
         vect = tfidf.transform(data).toarray()
         my_prediction = clf.predict(vect)
       return render_template('result.html',prediction = my_prediction) 

if __name__ == '__main__':
     app.run(debug=True, use_reloader=False)  
