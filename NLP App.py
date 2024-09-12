import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import spacy

nlp = spacy.load("en_core_web_sm")

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

st.title("Natural Language Processing Interface")

st.markdown('''
            <style>
            .reportview-container {
                background: white;
            }
            </style>'''
            , unsafe_allow_html=True)

text = st.text_area("Enter text", height=200, placeholder="Type your text here...")

options = st.selectbox("Select an option", [
    "Lemmatize",
    "Stem",
    "Tokenize",
    "Morphological analyze",
    "Word Generate",
    "Chunking",
    "Stop Word Removal",
    "Named Entity Recognition"
])

if options == "Lemmatize":
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]
    st.write("Lemmatized text:", lemmas)

elif options == "Stem":
    stemmer = PorterStemmer()
    stems = [stemmer.stem(word) for word in word_tokenize(text)]
    st.write("Stemmed text:", stems)

elif options == "Tokenize":
    tokens = word_tokenize(text)
    st.write("Tokenized text:", tokens)

elif options == "Morphological analyze":
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    st.write("Part-of-speech tags:", pos_tags)

elif options == "Word Generate":
    synsets = [wordnet.synsets(word) for word in word_tokenize(text)]
    generated_words = []
    for synset in synsets:
        if synset:
            generated_words.extend([lemma.name() for lemma in synset[0].lemmas()])
    st.write("Generated words:", generated_words)

elif options == "Chunking":
    doc = nlp(text)
    chunks = [(chunk.text, chunk.label_) for chunk in doc.noun_chunks]
    st.write("Chunks:", chunks)

elif options == "Stop Word Removal":
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in word_tokenize(text) if word.lower() not in stop_words]
    st.write("Text without stop words:", tokens)

elif options == "Named Entity Recognition":
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    st.write("Named entities:", entities)