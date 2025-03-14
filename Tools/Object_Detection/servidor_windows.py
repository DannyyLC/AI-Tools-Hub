import cv2

# Dirección RTSP desde el servidor FFmpeg en Windows
rtsp_url = 'rtsp://localhost:8554/mystream'

# Abre el stream RTSP usando OpenCV
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("No se pudo abrir el stream RTSP")
    exit()

# Lee un frame del stream
ret, frame = cap.read()

if ret:
    # Guarda la imagen capturada en el disco
    cv2.imwrite('captured_image.jpg', frame)
    print("Imagen guardada como captured_image.jpg")
else:
    print("No se pudo capturar la imagen")

# Libera el recurso
cap.release()
