from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import requests, re
import dominate
# pyright: reportWildcardImportFromLibrary=false
from dominate.tags import *

from scraper.page_processor import PageProcessor
from coherence.entities.coreferences import coref_annotation, coref_diagram
from tem.process import get_topic_evolution
from tem.nlp import docs_from_period, merge_short_periods
from stylometry.logistic_regression import predict_author
from train_tem_metrics import predict_from_tem_metrics


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
            te_prediction, te = run_tem_on(content)
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
        "Please note that this estimation does not need to be correct and should be further supported by the in-depth "
        "analysis below.")
    st.subheader("Topic Evolution Analysis:")
    image = te.graph().pipe(format='jpg')
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


def run_tem_on(text):
    corpus = [docs_from_period(line) for line in text.split('\n') if len(line) > 0]
    corpus = merge_short_periods(corpus, min_docs=2)
    te = get_topic_evolution(
        corpus,
        c=0.5,
        alpha=0,
        beta=-1,
        gamma=0,
        delta=1,
        theta=2,
        mergeThreshold=100,
        evolutionThreshold=100
    )
    return predict_from_tem_metrics(te), te


def get_prediction(style_prediction, te_prediction):
    te_confidence = te_prediction[1]
    te_prediction = te_prediction[0]
    if te_prediction == 0:
        te_prediction = -1
    style = sum(style_prediction)
    if style * te_prediction > 0:
        return te_prediction
    if style == 0 or te_confidence > 0.8:
        return te_prediction
    elif style < 0:
        if te_prediction < 0 or te_confidence < 0.6:
            return -1
        else:
            return 0
    else:
        if te_prediction < 0 and te_confidence > 0.7:
            return 0
        else:
            return 1


if __name__ == '__main__':
    col1, col2 = st.columns([3, 1])
    col1.title("Welcome at unBlock")
    col2.image(Image.open("unBlock.png"), width=100)
    st.write(
        " \nHere you can analyze a news article on topics and writing style to get further insights on whether this text "
        "might have been written by an AI. This system was developed at Hasso-Plattner-Institute. To start, please choose "
        "the type of input and enter the url/text in the field below.")
    col3, col4 = st.columns(2)
    input_type = col3.selectbox("type of input", ('URL', 'Text'), label_visibility="collapsed")
    text = ""
    if input_type == 'URL':
        text = st.text_input("URL to analyze:", "")
    else:
        text = st.text_area("Full text to analyze:", height=300)
    if col4.button("Compute Results"):
        run_analysis(input_type, text)
