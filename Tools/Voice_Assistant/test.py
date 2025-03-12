import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Cargar variables de entorno desde .env (si lo usas)
load_dotenv()

# Configurar el modelo de OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Pregunta simple
query = "Define que es el machine learning"

# Obtener respuesta del modelo
response = llm.invoke(query)

# Imprimir respuesta
print("Respuesta:", response)
