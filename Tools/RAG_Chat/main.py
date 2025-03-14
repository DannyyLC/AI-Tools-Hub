import asyncio
import os
from io import BytesIO
from src.indexing import EmbeddingProcessor
from src.shared.logging_utils import get_logger
from typing import Dict, Any
from langchain_core.messages import HumanMessage


logger = get_logger(__name__)


async def process_query(graph, state):
    """Maneja la opción de consulta al sistema."""
    print("\n=== Modo de Consulta ===")
    query = input("Ingresa tu pregunta: ")
    
    try:
        pass
        
    except Exception as e:
        logger.error(f"Error al procesar la consulta: {str(e)}")
        print(f"Error: {str(e)}")

async def index_documents(embedding_processor: EmbeddingProcessor):
    """Maneja la opción de indexación de documentos."""
    pdf_path = input("Dame el nombre del archivo: ")
    collection = "info_general"

    # Leer el archivo PDF
    with open(pdf_path, 'rb') as pdf_file:
        pdf_content = pdf_file.read()
    
    # Crear el BytesIO con el contenido del PDF
    pdf_bytes = BytesIO(pdf_content)
    
    # Obtener el nombre del archivo del path
    pdf_name = os.path.basename(pdf_path)
    
    # Crear la lista de documentos
    documents = [(pdf_bytes, pdf_name)]
    
    # Procesar y almacenar embeddings
    collection_name = await embedding_processor.process_and_store(
        documents=documents,
        user_id="usuario_123",
        collection_name=collection
    )
    
    print(f"Embeddings almacenados en la colección: {collection_name}")
    
async def main():
    """Función principal que muestra el menú y maneja las opciones."""
    # Instancias
    embedding_processor = EmbeddingProcessor(persist_directory="./chroma_db")


    
    while True:
        print("\n=== Sistema de Investigación ===")
        print("1. Indexar documentos")
        print("2. Realizar consulta")
        print("3. Salir")
        
        choice = input("Selecciona una opción (1-3): ")
        
        if choice == "1":
            await index_documents(embedding_processor)
        elif choice == "2":
            pass
        elif choice == "3":
            print("Saliendo del sistema. ¡Hasta pronto!")
            break
        else:
            print("Opción no válida. Por favor, intenta de nuevo.")

if __name__ == "__main__":
    # Ejecutar el bucle principal de forma asíncrona
    asyncio.run(main())