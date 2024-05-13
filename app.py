import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import string

ps = PorterStemmer()

st.set_page_config(page_title='Email/SMS Spam Classifier', layout='wide')


tfidf = pickle.load(open('tdidf_vecotrizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def preprocessing(text):
    text = text.lower()

    for txt in text:
        if txt in string.punctuation:
            text = text.replace(txt, "").strip()

    return text


def remove_stopwords(text):
    stop_words = stopwords.words('english')
    keep_words = []
    for words in text.split():
        if words not in stop_words:
            keep_words.append(words)
    return ' '.join(keep_words)


def steaming(text):
    text = nltk.word_tokenize(text)

    lst = []
    for i in text:
        lst.append(ps.stem(i))
    return ' '.join(lst)


page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://cdn.wallpapersafari.com/88/75/cLUQqJ.jpg");
  background-size: cover;
}
[data-testid="stHeader"]{
  background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_element, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black';>Email/SMS spam classifier 💬</h1>", unsafe_allow_html=True)

input_msg = st.text_area(placeholder="Enter the message",label="")

if st.button('Predict'):
    message = preprocessing(remove_stopwords(steaming(input_msg)))
    # message = transform_text(input_msg)

    vector_transform = tfidf.transform([message])
    result = model.predict(vector_transform)[0]
    with st.spinner('Wait for it'):
        time.sleep(1)
        if result == 1:
            st.markdown("<h1 style='text-align: center; color: black';>Spam</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center; color: black';>Not Spam</h1>", unsafe_allow_html=True)
            st.balloons()
