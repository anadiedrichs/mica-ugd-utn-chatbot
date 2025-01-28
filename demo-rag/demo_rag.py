# Importar las bibliotecas necesarias
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# 1. Cargar el modelo de embeddings en español
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Lista de documentos en español (puedes cargarlos desde una base de datos o archivo)
documents = [
    "El cambio climático es un problema global.",
    "La inteligencia artificial está transformando la industria.",
    "Python es un lenguaje de programación muy popular.",
    # Agrega más documentos aquí
]

# 3. Generar embeddings para los documentos y crear un índice FAISS
document_embeddings = embedder.encode(documents)
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# 4. Función para buscar documentos relevantes en FAISS
def search(query, top_k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]

# 5. Cargar el modelo de question-answering en español con autenticación
# leer el token de Hugging Face desde la variable de entorno
# os.getenv("HUGGINGFACE_TOKEN")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
qa_pipeline = pipeline(
    'question-answering', 
    model='mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es',
    tokenizer=(
        'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es',  
        {"use_fast": False}
    ),
    use_auth_token=HUGGINGFACE_TOKEN  # Autenticación con token
)

# 6. Función para obtener la respuesta usando el modelo de question-answering
def get_answer(question, context):
    result = qa_pipeline(question=question, context=context)
    return result

# 7. Función principal RAG que integra la búsqueda y la generación de respuestas
def rag(query, top_k=3):
    # Paso 1: Buscar los documentos más relevantes
    relevant_docs = search(query, top_k)
    
    # Paso 2: Usar el documento más relevante como contexto
    context = relevant_docs[0][0]
    
    # Paso 3: Obtener la respuesta usando el modelo de question-answering
    answer = get_answer(query, context)
    
    return answer

# 8. Ejemplo de uso del sistema RAG
if __name__ == "__main__":
    query = "¿Qué es el cambio climático?"
    answer = rag(query)
    print(f"Pregunta: {query}")
    print(f"Respuesta: {answer}")