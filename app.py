# from flask import Flask, render_template, request
import jsonify
import requests
import time
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from keras.models import load_model
import pandas as pd 
import numpy as np 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

def predict(link):
    
        data=[]
        with Chrome(executable_path=r'/home/unknown/Documents/chromedriver') as driver:
            wait = WebDriverWait(driver,15)
            driver.get(link)
            time.sleep(5)
            driver.execute_script('window.scrollTo(1, 100);')
            time.sleep(5)
            for item in range(20): 
                wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
                time.sleep(2)

            for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text"))):
                data.append(comment.text)

            df = pd.DataFrame(data, columns=['Comment'])
            df.drop(index=[0,1,2], axis=0, inplace=True)
            df["length"] = [len(i) for i in df["Comment"]]
            df1=df[df["length"] > 300].index
            df.drop(df1,axis=0,inplace=True)
            stopwords = set(nltk.corpus.stopwords.words('english'))
            vocab_size=10000
            len_sentence=150
            def text_prepare(data, column):
                print(data.shape)
                stemmer = PorterStemmer()
                corpus = []
                
                for text in data[column]:
                    
                    text = re.sub("[^a-zA-Z]", " ", text)
                    
                    text = text.lower()
                    text = text.split()
                    
                    text = [stemmer.stem(word) for word in text if word not in stopwords]
                    text = " ".join(text)
                    
                    corpus.append(text)
                one_hot_word = [one_hot(input_text=word, n=vocab_size) for word in corpus]
                embeddec_doc = pad_sequences(sequences=one_hot_word,
                                        maxlen=len_sentence,
                                        padding="pre")
                print(data.shape)
                return embeddec_doc
            emo=text_prepare(df, "Comment")
            kk= model.predict(emo)
            output=np.argmax(kk, axis=1)
            output=output.tolist()
            e={0:"anger",1:"fear",2:"joy",3:"love",4:"sadness",5:"surprise"}
            fuck=[]
            for i in range(6):
               fuck.append(int(output.count(i)*100/len(output)))

            return fuck

st.header('uthoob')
model = load_model('model.h5')
with st.form("my_form"):
    st.write("Youtube Link: ")
    Link = st.text_input("Paste youtube link here")
    if st.form_submit_button("Submit"):
        fuck = predict(Link)
        st.header('Results:')
        emoo=["anger","fear","joy","love","sadness","surprise"]
        
        for i in range(len(fuck)):
            with st.container():
                st.write(fuck[i],'% of comments are of emotion ', emoo[i])

# app = Flask(__name__)
# model = load_model('model.h5')

# @app.route('/',methods=['GET'])
# def Home():
#     return render_template('index.html')


# @app.route("/predict", methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         link = request.form['Link']
#         data=[]
#         with Chrome(executable_path=r'/home/unknown/Documents/chromedriver') as driver:
#             wait = WebDriverWait(driver,15)
#             driver.get(link)
#             time.sleep(5)
#             driver.execute_script('window.scrollTo(1, 100);')
#             time.sleep(5)
#             for item in range(20): 
#                 wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
#                 time.sleep(2)

#             for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text"))):
#                 data.append(comment.text)

#             df = pd.DataFrame(data, columns=['Comment'])
#             df.drop(index=[0,1,2], axis=0, inplace=True)
#             df["length"] = [len(i) for i in df["Comment"]]
#             df1=df[df["length"] > 300].index
#             df.drop(df1,axis=0,inplace=True)
#             stopwords = set(nltk.corpus.stopwords.words('english'))
#             vocab_size=10000
#             len_sentence=150
#             def text_prepare(data, column):
#                 print(data.shape)
#                 stemmer = PorterStemmer()
#                 corpus = []
                
#                 for text in data[column]:
                    
#                     text = re.sub("[^a-zA-Z]", " ", text)
                    
#                     text = text.lower()
#                     text = text.split()
                    
#                     text = [stemmer.stem(word) for word in text if word not in stopwords]
#                     text = " ".join(text)
                    
#                     corpus.append(text)
#                 one_hot_word = [one_hot(input_text=word, n=vocab_size) for word in corpus]
#                 embeddec_doc = pad_sequences(sequences=one_hot_word,
#                                         maxlen=len_sentence,
#                                         padding="pre")
#                 print(data.shape)
#                 return embeddec_doc
#             emo=text_prepare(df, "Comment")
#             kk= model.predict(emo)
#             output=np.argmax(kk, axis=1)
#             output=output.tolist()
#             e={0:"anger",1:"fear",2:"joy",3:"love",4:"sadness",5:"surprise"}
#             fuck=[]
#             for i in range(6):
#                fuck.append(int(output.count(i)*100/len(output)))

#             return render_template('index.html',prediction_text='''{} % of the comments are of emotion anger,\n 
#             {} % of the comments are of emotion fear,\n
#             {} % of the comments are of emotion joy,\n
#             {} % of the comments are of emotion love,\n
#             {} % of the comments are of emotion sadness,\n
#             {} % of the comments are of emotion surprise,\n
#             '''.format(fuck[0],fuck[1],fuck[2],fuck[3],fuck[4],fuck[5]))

# if __name__=="__main__":
#     app.run(debug=True)
