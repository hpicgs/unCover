from PIL import Image
import dominate
from dominate.tags import *
import requests
import threading
import subprocess
import sys
import os
import streamlit as st
import streamlit.components.v1 as components

from misc.entity_coreferences import coref_annotation, coref_diagram
from data_creation.page_processor import PageProcessor
from stylometry.corenlp import connect_corenlp
from stylometry.classifier import predict_author
from misc.tem_helpers import get_te_graph, get_tegm
from misc.tegm_training import predict_from_tegm
from definitions import ROOT_DIR


def __models_thread():
    proc = subprocess.run([os.path.join(ROOT_DIR, 'deployment', 'prepare_models')], capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout.decode())
    # we ignore exit code 2 because we might be waiting for another process to
    # finish the download
    if proc.returncode == 1:
        sys.exit(1)


def load_from_url(url):
    page = requests.get(url).text
    processor = PageProcessor(page)
    processed_page = processor.get_fulltext(separator="\n")
    print(processed_page)
    return processed_page


def run_analysis(input_type, user_input):
    if input_type == 'URL':
        content = load_from_url(user_input)
    else:
        content = user_input

    with st.spinner("Computing Analysis... for long texts this can take a few minutes"):

        style_prediction = predict_author(content)

        try:
            tegm = get_tegm([content])
            te_prediction = predict_from_tegm(tegm)
        except AttributeError:  # some texts are not working for tem
            st.error("The input text is too short for the Topic Evolution Model to work. Please enter a different "
                     "text. If you are using a URL, please try to copy the text manually since some websites can block "
                     "our scraper. And result in this error since no text was found.")
            return

        entity_html = entity_occurrence_diagram(content)

    author = get_prediction(style_prediction, te_prediction)
    if author == 1:
        st.subheader("This text was likely written by a machine!")
    elif author == -1:
        st.subheader("This text was likely written by a human author.")
    elif author == 0:
        st.subheader("We are not sure if this text was written by a machine or a human.")
    st.write(
        "Stylometry indicated that the text " + ("author could not be identified." if sum(style_prediction) == 0
                                               else "was written by a " + ("machine." if sum(style_prediction) > 0
                                                                           else "human.")))
    st.write(
        "Metrics on the Topic Graph indicated that the text was written by a " + ("machine, " if te_prediction[0] == 1
                                                                                  else "human, ")
        + f"with a confidence of {round(te_prediction[1] * 100, 2)}%.")
    st.write(
        "Please note that this estimation does not need to be correct and should be further supported by the in-depth "
        "analysis below.")
    st.subheader("Topic Evolution Analysis:")
    image = get_te_graph(content).pipe(format='jpg')
    st.image(image, caption="Topic Evolution on Input Text")
    st.subheader("Entity Occurrences Analysis:")
    components.html(entity_html, height=1000, scrolling=True)


def entity_occurrence_diagram(text):
    chart, legend = coref_diagram(coref_annotation(text))
    doc = dominate.document(title="Entity Occurrences")
    with doc:
        container = div(style='max-width: 900px; margin: auto')
        container.add(chart)
        container.add(h2('Legend'))
        container.add(legend)
    return doc.render()


def get_prediction(style_prediction, te_prediction):
    te_confidence = te_prediction[1]
    te_prediction = te_prediction[0]
    individual_styles = sum(style_prediction[:-1])
    style = style_prediction[3]
    if individual_styles * style < 0:
        return 0
    if style * te_prediction > 0:
        return te_prediction
    if style == 0 or te_confidence > 0.8:
        return te_prediction
    elif style < 0:
        if te_prediction <= 0 or te_confidence < 0.6:
            return -1
        else:
            return 0
    else:
        if te_prediction < 0 and te_confidence > 0.7:
            return 0
        else:
            return 1


if __name__ == '__main__':
    connect_corenlp()
    # ensure models are available
    thread = threading.Thread(target=__models_thread)
    thread.start()

    col1, col2 = st.columns([3, 1])
    col1.title("Welcome at unCover")
    col2.image(Image.open('./.streamlit/unCover.png'), width=100)
    st.write(
        " \nHere you can analyze a news article on topics and writing style to get further insights on whether this "
        "text might have been written by an AI. This system was developed at Hasso-Plattner-Institute. For more "
        "information and the associated paper visit https://github.com/hpicgs/unCover.")
    st.write("To start, please choose the type of input and enter the url/text in the field below.")
    col3, col4 = st.columns(2)
    input_type = col3.selectbox("type of input", ('URL', 'Text'), label_visibility='collapsed')
    text = ''
    if input_type == 'URL':
        text = st.text_input("URL to analyze:", '')
    else:
        text = st.text_area("Full text to analyze:", height=300)
    if col4.button("Compute Results"):
        run_analysis(input_type, text)
