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
def generate_embeddings(text, model):
    text_fragments = [text[i:i+1000] for i in range(0, len(text), 1000)]  # Dividir texto en fragmentos de 1000 caracteres
    embeddings = model.encode(text_fragments, show_progress_bar=True)
    return text_fragments, np.array(embeddings).astype("float32")

# Función para crear base de datos vectorial FAISS
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Dimensión de los embeddings
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Agregar embeddings al índice
    return index

# Función para responder preguntas (generativa)
def answer_question(question, faiss_index, text_fragments, embedding_model, generative_model):
    # Generar embedding para la pregunta
    question_embedding = np.array(embedding_model.encode([question])).astype("float32")
    distances, indices = faiss_index.search(question_embedding, k=5)

    # Obtener contexto relevante
    relevant_context = "\n\n".join([text_fragments[i] for i in indices[0]])

    # Usar el modelo generativo
    prompt = f"Eres un asistente experto en responder preguntas basadas en documentos PDF. Proporciona respuestas detalladas y explicativas.\n\nContexto: {relevant_context}\n\nPregunta: {question}\n\nRespuesta:"
    response = generative_model(prompt, max_length=512, num_return_sequences=1)[0]['generated_text']

    # Retornar la respuesta y el contexto relevante
    return response, relevant_context

# Inicialización del modelo de embeddings
embedding_model = SentenceTransformer("sentence-transformers/LaBSE")

# Inicialización del modelo generativo
generative_model = pipeline("text2text-generation", model="meta-llama/Llama-2-7b-chat-hf")

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
        text_fragments, embeddings = generate_embeddings(document_text, embedding_model)
        faiss_index = create_faiss_index(embeddings)
    st.success("Embeddings generados y almacenados en la base de datos vectorial.")

    # Paso 3: Responder preguntas
    question = st.text_input("Haz una pregunta sobre el documento:")
    if question:
        with st.spinner("Buscando respuesta..."):
            response, relevant_context = answer_question(question, faiss_index, text_fragments, embedding_model, generative_model)

        if response:
            st.write(f"**Respuesta:** {response}")
        else:
            st.write("No se encontró una respuesta adecuada.")

        st.write(f"**Contexto relevante:** {relevant_context}")
