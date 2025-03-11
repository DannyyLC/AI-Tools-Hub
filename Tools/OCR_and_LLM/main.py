from ocr_service import OCRService
from llm_service import LLMService

# Inicializar OCR y LLM
ocr = OCRService(language="en")
llm = LLMService(model="llama3.2:1b")

# Ruta de la imagen de prueba
image_path = "images/test1.png"

# Extraer texto de la imagen
extracted_text = ocr.extract_text(image_path)
print(f"Texto extraído:\n{extracted_text}")

# Pregunta del usuario sobre el texto extraído
user_prompt = "Retorname exactamente el mismo texto que te envie pero corrige la ortografia asi como pon las palabras faltantes si es necesario"

# Consultar al LLM con el texto extraído
response = llm.query_llm(extracted_text, user_prompt)
print("\nRespuesta del LLM:\n", response)
