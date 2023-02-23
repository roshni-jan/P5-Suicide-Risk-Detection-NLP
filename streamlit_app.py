
import contractions
import corpora
import dill as pickle
import joblib
import re 
import string
from nrclex import NRCLex
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, RobustScaler
import streamlit as st
import subprocess
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

python3 -m textblob.download_corpora

@st.cache
def load_model():
	final_pipe = joblib.load('final_pipe.joblib')
	return final_pipe

st.markdown(" ### NOTE:")
st.markdown("This is not meant to be a clinical or diagnostic tool. If you are experiencing thoughts of suicide, **help is available.**")
st.markdown("#### For immediate support, **call 988** or click [here](https://988lifeline.org/chat/)")
st.markdown("[Online Resources and Mental Health Apps](https://psychiatry.ucsf.edu/copingresources/apps)")
st.markdown("---")

st.header("Risk Detection Analyzer")
st.markdown(" ###### Test how the Suicide Risk Detection Model would classify different messages")
st.markdown("    ")
st.markdown("Type your message below and press **Submit**")



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

	df = pd.DataFrame([{'text':text} | sia.polarity_scores(text) | emot_dict])
	df = pd.concat([df,df])
	df.columns = ['text', 'neg_s', 'neut_s', 'pos_s', 'comp_s', 'anger_e', 'disgust_e', 'fear_e', 'sad_e', \
	'anticip_e', 'joy_e', 'surpr_e', 'trust_e']
	return df

def return_prediction(df, finalmodel):
	y_pred = np.where(finalmodel.predict_proba(textdata)[1,1] > 0.400, 1, 0)
    
	if y_pred == 1:
		return "Possible Suicide Risk indicated from text"
	if y_pred == 0:
		return "No Suicide Risk indicated from text"

if submit:
	cleantext = express_clean(st.session_state.user_text)
	textdata = get_data_from_text(cleantext)
	model = load_model()
	st.write(return_prediction(textdata, model))
	




