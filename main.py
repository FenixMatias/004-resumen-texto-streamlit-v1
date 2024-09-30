import streamlit as st
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generate_response(txt):
    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_api_key
    )
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )
    return chain.run(docs)

st.set_page_config(
    page_title = "Redacción de resúmenes de textos"
)
st.title("Redacción de resúmenes de textos")

st.write("Contacte con [Matias Toro Labra](https://www.linkedin.com/in/luis-matias-toro-labra-b4074121b/) para construir sus proyectos de IA")

txt_input = st.text_area(
    "Introduzca su texto",
    "",
    height=200
)

result = []
with st.form("summarize_form", clear_on_submit=True):
    openai_api_key = st.text_input(
        "Clave API de OpenAI",
        type="password",
        disabled=not txt_input
    )
    submitted = st.form_submit_button("Enviar")
    if submitted and openai_api_key.startswith("sk-"):
        response = generate_response(txt_input)
        result.append(response)
        del openai_api_key

translated_summary = f"Por favor, traduce el siguiente texto al español: {response}"

if len(result):
    st.info(translated_summary)