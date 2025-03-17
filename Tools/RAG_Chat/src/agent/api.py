import enum
import os
from typing import Annotated
from livekit.agents import llm
import logging
import torch
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from livekit.plugins import openai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("temperature-control")
logger.setLevel(logging.INFO)
load_dotenv()

class Zone(enum.Enum):
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"


class AssistantFnc(llm.FunctionContext):
    def __init__(self) -> None:
        super().__init__()

        self._temperature = {
            Zone.LIVING_ROOM: 22,
            Zone.BEDROOM: 20,
            Zone.KITCHEN: 24,
            Zone.BATHROOM: 23,
            Zone.OFFICE: 21,
        }
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configurar el modelo de embeddings
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        
        self.db = Chroma(
            persist_directory="./chroma_db",
            collection_name="info_general",
            embedding_function=self.embeddings
        )

    @llm.ai_callable(description="get the temperature in a specific room")
    def get_temperature(
        self, zone: Annotated[Zone, llm.TypeInfo(description="The specific zone")]
    ):
        logger.info("get temp - zone %s", zone)
        temp = self._temperature[Zone(zone)]
        return f"The temperature in the {zone} is {temp}C"

    @llm.ai_callable(description="set the temperature in a specific room")
    def set_temperature(
        self,
        zone: Annotated[Zone, llm.TypeInfo(description="The specific zone")],
        temp: Annotated[int, llm.TypeInfo(description="The temperature to set")],
    ):
        logger.info("set temp - zone %s, temp: %s", zone, temp)
        self._temperature[Zone(zone)] = temp
        return f"The temperature in the {zone} is now {temp}C"
    
    @llm.ai_callable(description="get information from the vector database to provide more accurate answers")
    def retrieval(
        self,
        query: Annotated[str, llm.TypeInfo(description="The query that you want to investigate in the vector data base")]
    ):
        logger.info("Extrallendo informacion del retrieval")
        logger.info("Query: " + query)
        results = self.db.similarity_search(query=query, k=3)
        if not results:
            return "I couldn't find any relevant information in the database."
        
         # Formatear los resultados en un solo texto
        context = "\n\n".join([res.page_content for res in results])

        # Crear una consulta estructurada para el LLM
        prompt = f"""Based on the following retrieved information, answer the user's query concisely and accurately.

        Retrieved information:
        {context}

        User query: {query}
        """
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        # Configurar el modelo de chat OpenAI con la clave de API cargada
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
        
        # Crear un mensaje inicial para la conversación
        messages = [{"role": "system", "content": "You are an assistant that should respond the user questions based on the retrieved information. give short answers, be precise and confirm your answers before giving me a result"}]
        messages.append({"role": "user", "content": prompt})
        
        # Usar el método 'invoke' para hacer una consulta
        response = llm.invoke(messages)
        logger.info("LLM response: " + response.content)
        return response.content