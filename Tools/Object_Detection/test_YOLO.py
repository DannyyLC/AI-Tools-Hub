import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Run inference on an image
results = model("data/koenigsegg.jpg")

# Convert results to an image with bounding boxes
annotated_frame = results[0].plot()

# Save the image
output_path = "data/output.jpg"
cv2.imwrite(output_path, annotated_frame)
print(f"Image saved at: {output_path}")
