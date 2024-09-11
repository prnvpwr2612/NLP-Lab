import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag, ne_chunk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('punkt')

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
    pos_tags = pos_tag(word_tokenize(text))
    st.write("Part-of-speech tags:", pos_tags)

elif options == "Word Generate":
    synsets = [wordnet.synsets(word) for word in word_tokenize(text)]
    generated_words = []
    for synset in synsets:
        if synset:
            generated_words.extend([lemma.name() for lemma in synset[0].lemmas()])
    st.write("Generated words:", generated_words)

elif options == "Chunking":
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    st.write("Chunked text:", chunked)

elif options == "Stop Word Removal":
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in word_tokenize(text) if word.lower() not in stop_words]
    st.write("Text without stop words:", tokens)

elif options == "Named Entity Recognition":
    named_entities = ne_chunk(pos_tag(word_tokenize(text)))
    st.write("Named entities:", named_entities)