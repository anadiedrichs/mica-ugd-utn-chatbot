import os
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# Importamos la InferenceApi de huggingface_hub en lugar de pipeline
from huggingface_hub import HfApi, InferenceApi

# -------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------

def extract_text_from_pdf(pdf_file):
    """
    Extrae y retorna el texto completo de un archivo PDF.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def generate_embeddings(text, model):
    """
    Genera los embeddings para fragmentos de texto de 1000 caracteres.
    Retorna (text_fragments, embeddings).
    """
    # Dividimos el texto en trozos de 1000 caracteres
    text_fragments = [text[i : i + 1000] for i in range(0, len(text), 1000)]
    embeddings = model.encode(text_fragments, show_progress_bar=True)
    return text_fragments, np.array(embeddings).astype("float32")

def create_faiss_index(embeddings):
    """
    Crea y retorna un índice FAISS a partir de los embeddings dados.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# -------------------------------------------------------------------
# Función de respuesta con la Inference API
# -------------------------------------------------------------------
def answer_question(
    question,
    faiss_index,
    text_fragments,
    embedding_model,
    generative_api,  # Aquí llega nuestro InferenceApi
    top_k=5
):
    """
    Dada una pregunta:
      - Genera embedding para la pregunta
      - Busca top_k fragmentos más cercanos en FAISS
      - Llama al modelo generativo (Inference API) con un prompt
      - Retorna (respuesta, contexto_relevante)
    """
    # 1. Generar embedding para la pregunta
    question_embedding = np.array(embedding_model.encode([question])).astype("float32")
    distances, indices = faiss_index.search(question_embedding, k=top_k)

    # 2. Obtener contexto relevante
    relevant_context = "\n\n".join([text_fragments[i] for i in indices[0]])

    # 3. Construir el prompt
    prompt = (
        "Eres un asistente experto en responder preguntas basadas en documentos PDF. "
        "Proporciona respuestas detalladas y explicativas.\n\n"
        f"Contexto: {relevant_context}\n\n"
        f"Pregunta: {question}\n\n"
        "Respuesta:"
    )

    # 4. Llamar a la Inference API con parámetros para text-generation
    #    El output suele venir en forma de lista de dict con 'generated_text'
    try:
        response = generative_api( prompt)

        # A veces la respuesta es un string directo o una lista de dict; depende del endpoint
        # Por convención, si es "text-generation", la respuesta suele ser: [{"generated_text": "..."}]
        if isinstance(response, list) and "generated_text" in response[0]:
            answer = response[0]["generated_text"]
        elif isinstance(response, dict) and "generated_text" in response:
            answer = response["generated_text"]
        else:
            # Fallback: si la respuesta no viene en ese formato
            answer = str(response)
    except Exception as e:
        answer = f"Error llamando a la Inference API: {str(e)}"

    return answer, relevant_context

# -------------------------------------------------------------------
# Comienzo de la aplicación Streamlit
# -------------------------------------------------------------------

st.title("Chatbot RAG basado en PDF (con Inference API)")

# 1. Leer el token de Hugging Face desde la variable de entorno
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if huggingface_token is None:
    st.error("El token de Hugging Face no está configurado como variable de entorno. Configúralo y reinicia la aplicación.")
    st.stop()

# 2. Inicializar la API de Hugging Face con el token
try:
    hf_api = HfApi(token=huggingface_token)
    user_info = hf_api.whoami()
    st.success(f"Autenticación exitosa con Hugging Face. Usuario: {user_info['name']}")
except Exception as e:
    st.error(f"Error al autenticar con Hugging Face: {e}")
    st.stop()

# 3. Modelo de embeddings (igual que antes)
embedding_model = SentenceTransformer("sentence-transformers/LaBSE")

# 4. Modelo generativo usando la Inference API de Hugging Face
#    Ajusta 'repo_id' al nombre exacto de tu repositorio/modelo
#    Asegúrate de especificar "task='text-generation'" si es un modelo generativo
generative_api = InferenceApi(
    repo_id="meta-llama/Llama-3.1-8B",
    token=huggingface_token
)

# 5. Interfaz con Streamlit
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
            response, relevant_context = answer_question(
                question,
                faiss_index,
                text_fragments,
                embedding_model,
                generative_api,      # Pasamos la InferenceApi
                top_k=5
            )

        if response:
            st.write(f"**Respuesta:** {response}")
        else:
            st.write("No se encontró una respuesta adecuada.")

        st.write(f"**Contexto relevante:**\n{relevant_context}")
