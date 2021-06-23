from os import write
import streamlit as st
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

#from transformers import pipeline

header = st.beta_container()
body = st.beta_container()
summary_container = st.beta_container()

######################## Summarization code  ########################################


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(
                sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(rawtext, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    article = rawtext.split(". ")
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    #print("Summarize Text: \n", ". ".join(summarize_text))
    return summarize_text

# This was a trial for abstractive summarization using transformers which works well but too slow
# def abstractive(rawtext):
#    summarizer = pipeline("summarization")
#    summary = summarizer(rawtext, max_length=300,
#                         min_length=200, do_sample=False)
#    summ = summary[0]
#    return summ['summary_text']

######################## Frontend code  ##############################################


with header:
    st.title('Text Summarization using NLTK')

with body:
    st.header('Extractive Summarization')
    rawtext = st.text_area('Enter Text Here')

    sample_col, upload_col = st.beta_columns(2)
    sample_col.header('Or select a sample file from below')
    sample = sample_col.selectbox('Or select a sample file',
                                  ('kalam_speech.txt', 'Stocks_ FRI_ JUN _8.txt', 'microsoft.txt', 'None'), index=3)
    if sample != 'None':
        file = open(sample, "r", encoding= 'cp1252')
        #st.write(file)
        rawtext = file.read()

    upload_col.header('Or upload text file here')
    uploaded_file = upload_col.file_uploader(
        'Choose your .txt file', type="txt")
    if uploaded_file is not None:
        rawtext = str(uploaded_file.read(), 'cp1252')

    no_of_lines = st.slider("Select number of lines in summary", 1, 5, 3)
    if st.button('Get Summary'):
        with summary_container:
            if rawtext == "":
                st.header('Summary :)')
                st.write('Please enter text to see summary')
            else:
                result = generate_summary(rawtext, no_of_lines)
                st.header('Summary :)')
                for i in range(no_of_lines):
                    st.write(result[i])

                # Abstractive summary
                #st.header('Abstractive method')
                #abstract = abstractive(rawtext)
                # st.write(abstract)

                st.header('Actual article')
                st.write(rawtext)
