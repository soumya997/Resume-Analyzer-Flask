import re
import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import spacy
import fitz
from werkzeug.utils import secure_filename
import pickle
import nltk
import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pdb
# import logging
from werkzeug.debug import DebuggedApplication
nltk.download()


# UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
# ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True
ROOT_DIR = os.getcwd()
FileSaveDir = os.path.join(ROOT_DIR, "uploads")
# application = DebuggedApplication(app, True)

# DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/home')
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/summary',methods=["GET","POST"])
def get_summary():
    print("inside get_summary")
    if request.method == 'POST':
        f = request.files['file']
        FileSavePath = os.path.join(FileSaveDir, f.filename)
        f.save(FileSavePath)
        cus_ents = custom_NER(FileSavePath,f.filename)
        return render_template('index2.html',message=cus_ents)

    return render_template('index2.html')


@app.route('/score',methods=["GET","POST"])
def get_score():
    print("inside get_score")
    if request.method == 'POST':
        f = request.files['file']
        FileSavePath = os.path.join(FileSaveDir, f.filename)
        f.save(FileSavePath)
        result = prediction(FileSavePath,f.filename)
        print(request.form)
        result = int(result)+1
        return render_template('index3.html',message=f"your score is {result}")

    return render_template('index3.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'favicons'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')



def custom_NER(path,filename):
    model = spacy.load('resume_sum1')
    txt=red_pdf(path,filename)
    doc = model(txt)
    abc = doc.ents
    return abc


def red_pdf(path,filename):
    doc = fitz.open(path)
    text = ""

    for pages in doc:
        text = text + str(pages.getText())
    txt = " ".join(text.split("\n"))
    return txt



def gen_test_data_for_pred(path,filename):
    test = red_pdf(path,filename)
    snow = nltk.stem.SnowballStemmer('english')
    corpus_test = []
    # for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', test)
    review = review.lower()
    review = review.split()

    review = [snow.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_test.append(review)

    final_tf_test = corpus_test
    # tf_idf = TfidfVectorizer(ngram_range=(1,2),max_features=5000)
    tf_idf = pickle.load(open('data/tfidf_vectorizer.pkl','rb'))
    test_data = tf_idf.transform(final_tf_test)
    # tf_data_test.get_shape()
    return test_data


def prediction(path,filename):
    clf_model = pickle.load(open('data/rf_score_model.pkl','rb'))
    result = clf_model.predict(gen_test_data_for_pred(path,filename))
    return result





if __name__ == '__main__':
   app.run(debug=True)
