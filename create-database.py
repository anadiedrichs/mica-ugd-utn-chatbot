import os
import requests
import pandas as pd
from pypdf import PdfReader

# LangChain y otras utilidades
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# Vector stores
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma"
DATA_PATH = "data"
EMBED_MODEL = "hiiamsid/sentence_similarity_spanish_es"


class PDFManager:
    def __init__(self, data_path: str = DATA_PATH):
        """
        Inicializa el administrador de PDFs.
        
        :param data_path: Ruta donde se descargarán y almacenarán los PDF.
        """
        self.data_path = data_path
        self.df = None
        self.documents = []
        self.split_docs = []
        self.vector_store = None   # Aquí almacenaremos la instancia de FAISS

    def get_source(self) -> pd.DataFrame:
        """
        Crea y retorna un DataFrame con la información sobre los PDFs:
        títulos, links, nombres de archivo y páginas iniciales/finales.
        """
        links = [
            "https://editorial.mingeneros.gob.ar:8080/xmlui/bitstream/handle/123456789/32/Violencias%20por%20motivos%20de%20g%c3%a9nero%20-%20MMGyD.pdf",
            "https://editorial.mingeneros.gob.ar:8080/xmlui/bitstream/handle/123456789/26/Perspectiva%20de%20g%c3%a9nero%20y%20diversidad%20-%20MMGyD.pdf",
            "https://editorial.mingeneros.gob.ar:8080/xmlui/bitstream/handle/123456789/19/Masculinidades%20sin%20violencia%20-%20MMGyD.pdf",
            "https://editorial.mingeneros.gob.ar:8080/xmlui/bitstream/handle/123456789/18/Igualdad%20en%20los%20cuidados.pdf",
            "https://editorial.mingeneros.gob.ar:8080/xmlui/bitstream/handle/123456789/35/Diversidad%20-%20MMGyD.pdf"
        ]
        titles = [
            "Violencia por motivos de género",
            "Perspectiva de género y diversidad",
            "Masculinidades sin violencia",
            "Igualdad en los cuidados",
            "Diversidad una perspectiva para la igualdad"
        ]
        file_name = [
            "violencia.pdf",
            "perspectiva.pdf",
            "masculinidades.pdf",
            "cuidados.pdf",
            "diversidad.pdf"
        ]
        # Se restan 1 a las páginas iniciales porque el índice comienza en 0
        pag_from = [14-1, 14-1, 14-1, 14-1, 12-1]
        pag_to = [85, 97, 30, 98, 87]

        data = {
            "title": titles,
            "link": links,
            "file_name": file_name,
            "pag_from": pag_from,
            "pag_to": pag_to
        }

        self.df = pd.DataFrame(data)
        print(self.df)
        return self.df

    def download_files(self) -> None:
        """
        Descarga los archivos PDF especificados en el DataFrame
        y los guarda en la ruta especificada por data_path.
        """
        if self.df is None:
            raise ValueError("DataFrame no definido. Llama primero al método get_source().")

        os.makedirs(self.data_path, exist_ok=True)

        for index, row in self.df.iterrows():
            file_path = os.path.join(self.data_path, row["file_name"])
            url = row["link"]
            response = requests.get(url)
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Descargado: {file_path}")

    def load_documents(self) -> None:
        """
        Carga los documentos PDF entre las páginas definidas en el DataFrame,
        generando objetos Document con su texto y metadatos.
        """
        if self.df is None:
            raise ValueError("DataFrame no definido. Llama primero al método get_source().")

        for index, row in self.df.iterrows():
            file_path = os.path.join(self.data_path, row["file_name"])
            reader = PdfReader(file_path)
            
            # Obtenemos las páginas deseadas
            pages = reader.pages[row["pag_from"]:row["pag_to"]]

            # Convertimos cada página en Document, con metadatos
            for i, page in enumerate(pages, start=row["pag_from"] + 1):
                text_content = page.extract_text()
                if not text_content:
                    continue
                self.documents.append(
                    Document(
                        page_content=text_content,
                        metadata={
                            'page': i,
                            'source': row["file_name"],
                            'title': row["title"]
                        }
                    )
                )

        print(f"Total de documentos cargados: {len(self.documents)}")

    def remove_line_breaks(self) -> None:
        """
        Elimina la secuencia '-\n' en el contenido de cada página
        para facilitar la lectura y el procesamiento.
        """
        for doc in self.documents:
            doc.page_content = doc.page_content.replace("-\n", "")
        print("Saltos de línea '-\\n' removidos.")

    def split_documents(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[Document]:
        """
        Divide los documentos en fragmentos más manejables, utilizando
        RecursiveCharacterTextSplitter. Retorna la lista de documentos divididos.
        
        :param chunk_size: Tamaño de cada fragmento de texto.
        :param chunk_overlap: Número de caracteres que se solapan entre fragmentos consecutivos.
        :return: Lista de documentos divididos.
        """
        if not self.documents:
            raise ValueError("No hay documentos para dividir. Carga documentos antes de llamar a este método.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.split_docs = text_splitter.split_documents(self.documents)
        print(f"Documentos divididos en {len(self.split_docs)} fragmentos.")
        return self.split_docs

    def create_vector_database(
        self, 
        path: str, 
        embedding_model: str = EMBED_MODEL
    ) -> None:
        """
        Vectoriza los documentos ya fragmentados (self.split_docs) y crea 
        una base de datos vectorial FAISS, que se guarda localmente.

        :param path: Ruta donde se guardarán los datos de FAISS.
        :param embedding_model: Nombre del modelo de embeddings a usar.
        """
        if not self.split_docs:
            raise ValueError("No hay documentos fragmentados. Ejecuta split_documents() primero.")

        # Crear embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Crear la base de datos vectorial FAISS
        self.vector_store = FAISS.from_documents(self.split_docs, embeddings)
        
        # Guardar la base vectorial de FAISS en el path indicado
        self.vector_store.save_local(path)
        print(f"Base de datos vectorial FAISS creada y guardada en: {path}")

    def load_vector_database(
        self,
        path: str,
        embedding_model: str = EMBED_MODEL,
        allow_dangerous_deserialization: bool = False
    ) -> FAISS:
        """
        Carga una base de datos vectorial FAISS desde el path indicado,
        usando el mismo modelo de embeddings con el que se generó.

        :param path: Ruta desde donde se cargará la base de datos de FAISS.
        :param embedding_model: Nombre del modelo de embeddings a usar.
        :param allow_dangerous_deserialization: Si es True, permite deserializar el índice FAISS
                                               (pickle) que podría ser inseguro si proviene 
                                               de una fuente no confiable. Úsalo solo si
                                               confías en el archivo.
        :return: La instancia de la base vectorial FAISS cargada.
        """
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = FAISS.load_local(
            path, 
            embeddings, 
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        print(f"Base de datos vectorial FAISS cargada desde: {path}")
        return self.vector_store

    def search_vector_database(self, query: str, k: int = 3):
        """
        Realiza una búsqueda de similitud en la base de datos vectorial FAISS,
        retornando los documentos más relevantes para la consulta (query).
        
        :param query: Texto/prompt a buscar.
        :param k: Número de resultados a retornar.
        :return: Lista de objetos Document que coinciden más con la consulta.
        """
        if not self.vector_store:
            raise ValueError("No existe una base vectorial. Crea o carga una con create_vector_database o load_vector_database.")

        # Realiza la búsqueda de similitud
        results = self.vector_store.similarity_search(query, k=k)
        return results


def create_database(faiss_path: str = "faiss_index") -> PDFManager:
    """
    Función que encapsula todo el flujo de creación de la base de datos FAISS:
    1. Crea una instancia de PDFManager.
    2. Obtiene y descarga las fuentes.
    3. Carga y limpia documentos.
    4. Los divide en fragmentos.
    5. Crea la base de datos vectorial FAISS (y la guarda en faiss_path).

    Retorna la instancia de PDFManager para uso posterior.
    """
    pdf_manager = PDFManager()

    # 1. Obtener información fuente (DataFrame)
    pdf_manager.get_source()

    # 2. Descargar archivos PDF
    pdf_manager.download_files()

    # 3. Cargar documentos según páginas indicadas
    pdf_manager.load_documents()

    # 4. Eliminar saltos de línea '-\n'
    pdf_manager.remove_line_breaks()

    # 5. Dividir documentos en fragmentos
    pdf_manager.split_documents(chunk_size=800, chunk_overlap=80)

    # 6. Crear la base de datos vectorial FAISS y guardarla localmente
    pdf_manager.create_vector_database(path=faiss_path, embedding_model=EMBED_MODEL)

    return pdf_manager


if __name__ == "__main__":
    """
    Si 'faiss_index' existe, se carga la base vectorial FAISS. 
    Si no existe, se crea la base (lo que incluye descargar y procesar los PDFs).
    """
    faiss_path = "faiss_index"

    if os.path.exists(faiss_path):
        print(f"El path '{faiss_path}' ya existe. Se cargará la base de datos vectorial.")
        manager = PDFManager()
        manager.load_vector_database(path=faiss_path, 
                                     embedding_model=EMBED_MODEL,
                                     allow_dangerous_deserialization=True  # ÚSALO SOLO SI CONFÍAS EN EL ARCHIVO
                                     )
    else:
        print(f"El path '{faiss_path}' no existe. Creando la base de datos vectorial FAISS...")
        manager = create_database(faiss_path=faiss_path)

    # Ejemplo de uso de la función de búsqueda
    query = "¿Cuáles son los tipos de violenciad de género?"
    resultados = manager.search_vector_database(query, k=10)

    print("=== Resultados de la búsqueda ===")
    for idx, doc in enumerate(resultados, start=1):
        print(f"\nDocumento {idx}:\n{doc.page_content}\n\n(Metadata: {doc.metadata})")
