from langchain_ollama  import ChatOllama
from langchain.prompts import PromptTemplate

class LLMService:
    def __init__(self, model="mistral"):
        """Inicializa el LLM con el modelo especificado en Ollama."""
        self.llm = ChatOllama(model=model)

    def query_llm(self, text, user_prompt):
        """Envía el texto extraído al LLM junto con una solicitud del usuario."""
        prompt_template = PromptTemplate(
            template="Texto extraído: {text}\n\n{user_prompt}",
            input_variables=["text", "user_prompt"]
        )
        prompt = prompt_template.format(text=text, user_prompt=user_prompt)
        response = self.llm.invoke(prompt)
        return response
