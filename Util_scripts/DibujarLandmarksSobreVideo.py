import cv2
import os
import shutil
from easy_ViTPose import VitInference
import mediapipe as mp

# Configuración
video_path = r"D:\Miércoles varias\WhatsApp Video 2025-06-02 at 2.47.37 PM.mp4"  
output_video_path = r"D:\Miércoles varias\BackupIA\Video\VideoFace_yo_1.mp4"
temp_frames_dir = r"D:\Miércoles varias\BackupIA\Temp"

model_path = r"D:\Cosas varias\Proyecto IA\vitpose-b-wholebody.pth"
yolo_path = r"D:\Cosas varias\Proyecto IA\yolov8s.pt"
MODEL_SIZE = 'b'
DATASET = 'wholebody'

#  Inicializar modelos 
vit_model = VitInference(model_path, yolo_path, MODEL_SIZE, dataset=DATASET, yolo_size=320, is_video=True)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Crear carpeta temporal de frames 
if os.path.exists(temp_frames_dir):
    shutil.rmtree(temp_frames_dir)
os.makedirs(temp_frames_dir, exist_ok=True)

# Extraer frames del video original 
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f" Procesando {frame_count} frames a {fps} fps...")

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(temp_frames_dir, f"frame_{i:04d}.jpg")
    cv2.imwrite(frame_path, frame)

cap.release()


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


for i in range(frame_count):
    frame_path = os.path.join(temp_frames_dir, f"frame_{i:04d}.jpg")
    frame = cv2.imread(frame_path)

    # MediaPipe FaceMesh
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )

    # ViTPose
    vit_model.inference(frame)
    frame = vit_model.draw(show_yolo=False)

    out.write(frame)

out.release()
print(f"✅ Video final con posturas guardado en: {output_video_path}")

# === Limpieza opcional ===
shutil.rmtree(temp_frames_dir)
