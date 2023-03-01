from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import dominate
# pyright: reportWildcardImportFromLibrary=false
from dominate.tags import *

from coherence.entities.coreferences import coref_annotation, coref_diagram


def load_from_url(url):
    # TODO: load article
    return "TODO"


def run_analysis(m, user_input):
    if m == 'URL':
        content = load_from_url(user_input)
    else:
        content = user_input
    with st.spinner('Wait for it...'):
        # TODO: Inlcude other analyis in here!
        entity_html = entity_occurances(content)
    st.subheader("Entity Occurrences Analysis:")
    components.html(entity_html, height=1000, scrolling=True)


def entity_occurances(text):
    chart, legend = coref_diagram(coref_annotation(text))
    doc = dominate.document(title="Entity Occurrences")
    with doc:
        container = div(style='max-width: 900px; margin: auto')
        container.add(chart)
        container.add(h2('Legend'))
        container.add(legend)
    return doc.render()

if __name__ == '__main__':
    col1, col2 = st.columns([3, 1])
    col1.title("Welcome at unBlock")
    col2.image(Image.open("unBlock.png"), width=100)
    st.write(
        " \nHere you can analyze a news article on topics and writing style to get further insights on if this text "
        "might be written by an AI. This system was developed at Hasso-Plattner-Institute. To start please choose "
        "the type of input and enter the url/ text in the field below.")
    col3, col4 = st.columns(2)
    mode = col3.selectbox("type of input", ('URL', 'Text'), label_visibility="collapsed")
    text = ""
    if mode == 'URL':
        text = st.text_input("URL to analyze:", "")
    else:
        text = st.text_area("Full-text to analyze:", height=300)
    if col4.button("Compute Results"):
        run_analysis(mode, text)
