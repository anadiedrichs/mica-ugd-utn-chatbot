import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Función para generar embeddings
def generate_embeddings(text, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    text_fragments = text.split("\n\n")  # Dividir en párrafos
    embeddings = model.encode(text_fragments, show_progress_bar=True)
    return text_fragments, embeddings

# Función para crear base de datos vectorial FAISS
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Dimensión de los embeddings
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Agregar embeddings al índice
    return index

# Función para responder preguntas
def answer_question(question, faiss_index, text_fragments, model_name="distilbert-base-uncased-distilled-squad"):
    # Usamos el modelo para generar embeddings de la pregunta
    question_embedding = model.encode([question])
    distances, indices = faiss_index.search(np.array(question_embedding).astype("float32"), k=5)
    relevant_context = "\n\n".join([text_fragments[i] for i in indices[0]])

    # Usamos el modelo de preguntas y respuestas
    qa_pipeline = pipeline("question-answering", model=model_name)
    response = qa_pipeline(question=question, context=relevant_context)
    return response

# Interfaz con Streamlit
st.title("Chatbot RAG basado en PDF")
st.write("Este chatbot responde preguntas basadas en documentos PDF cargados por el usuario.")

uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")

if uploaded_file:
    # Paso 1: Extraer texto del PDF
    with st.spinner("Extrayendo texto del documento..."):
        document_text = extract_text_from_pdf(uploaded_file)
    st.success("Texto extraído.")
    st.text_area("Texto extraído (primeros 1000 caracteres):", document_text[:1000], height=250)

    # Paso 2: Generar embeddings
    with st.spinner("Generando embeddings del texto..."):
        text_fragments, embeddings = generate_embeddings(document_text)
        embeddings_np = np.array(embeddings).astype("float32")
        faiss_index = create_faiss_index(embeddings_np)
    st.success("Embeddings generados y almacenados en la base de datos vectorial.")

    # Paso 3: Responder preguntas
    question = st.text_input("Haz una pregunta sobre el documento:")
    if question:
        with st.spinner("Buscando respuesta..."):
            response = answer_question(question, faiss_index, text_fragments)
        st.write(f"**Respuesta:** {response['answer']}")
        st.write(f"**Contexto relevante:** {response['context']}")

