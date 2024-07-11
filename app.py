import streamlit as st
import pickle
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



ps = PorterStemmer()

tidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return ' '.join(y)


st.title('Email Spam Classifier')

input_email = st.text_area('Enter the message')

if st.button('Predict'):

    

    # 1. Preprocessing
    transform_email = transform_text(input_email)

    # 2. Vectorization
    vector_input = tidf.transform([transform_email])

    # 3. Prediction
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

