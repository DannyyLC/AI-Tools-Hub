import asyncio
import os
from io import BytesIO
from src.indexing.store_documents import EmbeddingProcessor

async def main():
    # Crear instancia del procesador
    embedding_processor = EmbeddingProcessor(persist_directory="./chroma_db")
    
    elementos = ["46", "50", "54", "68", "69", "71", "72", "73", "74", "75", "76", "77"]
    original = "OrdenDeVenta-358"
    
        
    # Ruta al archivo PDF
    pdf_path = "information/alumnos_calificaciones.xlsx" 
    print("Indexando: " + pdf_path)
    
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
        collection_name="info_general"
    )
    
    print(f"Embeddings almacenados en la colección: {collection_name}")

if __name__ == "__main__":
    asyncio.run(main())