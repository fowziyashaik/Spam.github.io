import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


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

    return " ".join(y)


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.markdown(
    """
    <h1 style="color: lightpink ">SMS Spam Detection Test</h1>
    """,
    unsafe_allow_html=True
)

    
st.markdown(
    """
    <h5 style="color:white; "> Enter the Email/Message text below to classify its Spam or Not Spam </h5>
    <h5 style="color: white; "> Inbox</h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[data-testid="stTextArea"] > div > textarea {
        height: 250px; /* Set height for 10 lines */
        font-size: 16px;
        padding: 10px;
        border: 2px solid #ccc;
        border-radius: 10px;
        width: 100%; /* Full width */
        box-sizing: border-box;
        background-color:lightgrey;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Multiline input field with a placeholder
input_sms = st.text_area("Enter the SMS", placeholder="Type your message here...")


if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tk.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
       
        st.markdown(
                """
                <div style="
                    color: white;
                    background: linear-gradient(to right, #ff4e50, #f9d423);
                    padding: 20px;
                    text-align: center;
                    border-radius: 15px;
                    box-shadow: 0 8px 4px rgba(0, 0, 0, 0.2);
                    font-size: 15px;
                    font-weight: normal;
                    transform: perspective(500px) rotateX(5deg);
                ">
                    🚫   Spam
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        
        st.markdown(
                """
                <div style="
                    color: white;
                    background: linear-gradient(to right, #56ab2f, #a8e063);
                    padding: 20px;
                    text-align: center;
                    border-radius: 15px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    font-size: 15px;
                    font-weight: normal;
                    transform: perspective(500px) rotateX(5deg);
                ">
                    ✅  Not  Spam
                </div>
                """,
                unsafe_allow_html=True,
            )
st.write("")
st.write("*Made by Shaik Fowziya *")
    
