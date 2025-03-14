import cv2
import torch
import time


# Verifica si CUDA está disponible y usa la GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ', device)

# Carga el modelo YOLOv5 (usando 'yolov5s' para mayor velocidad)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)

# Abre la cámara (ajusta el índice si no es la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Ajuste de resolución de la cámara
cap.set(3, 640)  # Ancho
cap.set(4, 480)  # Alto
cap.set(5, 15)   # FPS (ajusta según tu cámara)

# Limitar FPS (ejemplo: procesar solo 15 frames por segundo)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Limitar FPS a 15
    if time.time() - prev_time >= 1./15:  # 15 FPS
        prev_time = time.time()

        # Convierte el frame de BGR a RGB para YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realiza la detección de objetos en la imagen (usando CUDA si está disponible)
        results = model(frame_rgb)  # Esto usará la GPU si está disponible

        # Obtén el resultado de la detección
        frame_result = results.render()[0]  # Renderiza los resultados sobre el frame

        # Convierte el frame de vuelta a BGR para OpenCV
        frame_bgr = cv2.cvtColor(frame_result, cv2.COLOR_RGB2BGR)

        # Muestra el frame con las detecciones
        cv2.imshow('YOLO - Detección de objetos', frame_bgr)

    # Condición para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
