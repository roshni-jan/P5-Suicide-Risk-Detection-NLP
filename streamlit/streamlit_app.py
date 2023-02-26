import contractions
import dill as pickle
import joblib
from nrclex import NRCLex
import numpy as np
import pandas as pd
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer, RobustScaler
import streamlit as st
import string
import subprocess
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.markdown(" ### NOTE:")
st.markdown("This is not meant to be a clinical or diagnostic tool. If you are experiencing thoughts of suicide, **help is available.**")
st.markdown("#### For immediate support, **call 988** or click [here](https://988lifeline.org/chat/)")
st.markdown("[Online Resources and Mental Health Apps](https://psychiatry.ucsf.edu/copingresources/apps)")
st.markdown("---")

st.header("Risk Detection Analyzer")
st.markdown(" ###### Test how the Suicide Risk Detection Model would classify different messages")
st.markdown("    ")
st.markdown("Type your message below and press **Submit**")

cmd = ['python3','-m','textblob.download_corpora']
subprocess.run(cmd)
print("Working")


form = st.form(key='my_form')
form.text_input(label='Enter text here:', key='user_text')
submit = form.form_submit_button(label='Submit')

def express_clean(text):
	#  Remove HTML Characters and URLs
	text = text.encode("ascii", "ignore")
	text = text.decode()
	patterns = r'|'.join(map(r'(?:{})'.format, 
		(r"\n&\S+", r"\n", r"&lt", r"&gt", r"u/\S+", r"ww\S+", r"htt\S+", r"\d\S+", r"\d+")))
	text = re.sub(patterns, '', text)
	text = re.sub(r"(?<![A-Z\W])(?=[A-Z])", " ", text)
	#  Fix Contractions
	text = contractions.fix(text)
	#  Remove Punctuation
	translator = str.maketrans('', '', string.punctuation)
	text = text.translate(translator)
	# Make sure all characters are alphanumeric
	for word in text:
		if word.isalnum():
			continue
		else:
			for char in word:
				if not char.isalnum():
					word = word.replace(char, '')         
    # Lowercase and remove extra punctuation
	text = text.lower()
	text = text.strip()
	return text

def get_data_from_text(text):
	jl_tfidf = joblib.load('tfidf_vect.joblib')
	jl_rs = joblib.load('rs.joblib')
	jl_norm = joblib.load('norm.joblib')

	X_text_trf = pd.DataFrame(jl_tfidf.transform(pd.Series(text, index=[0])).toarray())

	sia = SentimentIntensityAnalyzer()
	emotion_list = ['anger', 'disgust', 'fear', 'sadness', 'anticipation', 'joy', 'surprise', 'trust']
	pop_list = ['positive', 'negative']
	emot_dict = NRCLex(text).raw_emotion_scores
	for key in pop_list:
		if key in emot_dict.keys():
			emot_dict.pop(key)
	for emot in emotion_list:
		if emot not in emot_dict.keys():
		 emot_dict[emot] = 0

	X_nums = pd.DataFrame([sia.polarity_scores(text) | emot_dict])
	col_order = ['neg', 'neu', 'pos', 'compound', 'anger', 'disgust', 'fear', 'sadness', 'anticipation', 'joy', 'surprise', 'trust']
	X_nums = X_nums[col_order]
	X_nums_sc = jl_rs.transform(X_nums)
	X_nums_sc = jl_norm.transform(X_nums_sc)
	X_nums_sc = pd.DataFrame(X_nums_sc)

	X_trf = pd.concat([X_text_trf, X_nums_sc], axis=1)
	return X_trf

def return_prediction(df):
	jl_logreg = joblib.load('logreg.joblib')
	y_pred = np.where(jl_logreg.predict_proba(textdata)[:,1] > 0.400, 1, 0)

	if y_pred == 1:
		return "Possible Suicide Risk indicated from text"
	if y_pred == 0:
		return "No Suicide Risk indicated from text"

if submit:
	cleantext = express_clean(st.session_state.user_text)
	textdata = get_data_from_text(cleantext)
	st.write(return_prediction(textdata))
	




