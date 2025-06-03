import cv2
import mediapipe as mp
import os
import shutil
from easy_ViTPose import VitInference

# Configuración
MODEL_SIZE = 'b'
DATASET = 'wholebody'
model_path = r"D:\Cosas varias\Proyecto IA\vitpose-b-wholebody.pth"
yolo_path = r"D:\Cosas varias\Proyecto IA\yolov8s.pt"
output_video_path = r"D:\Miércoles varias\BackupIA\Video\VideoConPosturas.mp4"
temp_frames_dir = r"D:\Miércoles varias\BackupIA\Temp"

# Crear carpeta temporal limpia
if os.path.exists(temp_frames_dir):
    shutil.rmtree(temp_frames_dir)
os.makedirs(temp_frames_dir)

# Fase 1: Grabación pura
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  # Asume grabación a 20 fps

print(" Grabando. Presiona ESC para detener...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Grabando video sin inferencias...', frame)
    frame_path = os.path.join(temp_frames_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ {frame_count} frames capturados y guardados en: {temp_frames_dir}")

# Fase 2: Procesamiento de Frames
print(" Procesando frames con MediaPipe y ViTPose...")

# Inicializar modelos
vit_model = VitInference(model_path, yolo_path, MODEL_SIZE, dataset=DATASET, yolo_size=320, is_video=True)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Inicializar escritor de video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Procesar cada frame
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

# Limpieza opcional
shutil.rmtree(temp_frames_dir)

