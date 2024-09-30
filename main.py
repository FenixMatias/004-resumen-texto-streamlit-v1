import streamlit as st
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generate_response(txt, openai_api_key):
    # Crea el modelo de lenguaje
    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_api_key
    )
    
    # Divide el texto en segmentos
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    
    # Cargar la cadena de resumen (map-reduce)
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )
    
    # Ejecuta la cadena para resumir
    summary = chain.run(docs)
    
    # Traduce el resumen al español
    translation_prompt = f"Por favor, traduce el siguiente texto al español: {summary}"
    translation = llm(translation_prompt)
    
    return translation

# Configurar la interfaz de Streamlit
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
        # Genera el resumen y la traducción
        response = generate_response(txt_input, openai_api_key)
        result.append(response)
        del openai_api_key

# Mostrar el resultado traducido
if len(result):
    st.info(result[0])