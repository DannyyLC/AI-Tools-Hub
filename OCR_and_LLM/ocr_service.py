import easyocr

class OCRService:
    def __init__(self, language="en"):
        """Inicializa el OCR con el idioma especificado."""
        self.reader = easyocr.Reader([language])
    
    def extract_text(self, image_path):
        """Extrae texto de una imagen y devuelve el resultado."""
        results = self.reader.readtext(image_path)
        extracted_text = " ".join([text for _, text, _ in results])
        return extracted_text
