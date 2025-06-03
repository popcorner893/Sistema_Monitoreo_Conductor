import cv2
import mediapipe as mp
from easy_ViTPose import VitInference

# 📌 Inicialización
MODEL_SIZE = 'b'
DATASET = 'wholebody'
model_path = r"D:\Cosas varias\Proyecto IA\vitpose-b-wholebody.pth"
yolo_path = r"D:\Cosas varias\Proyecto IA\yolov8s.pt"

vit_model = VitInference(model_path, yolo_path, MODEL_SIZE, dataset=DATASET, yolo_size=320, is_video=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 📌 MediaPipe FaceMesh
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

    # 📌 ViTPose
    vit_model.inference(frame)
    frame = vit_model.draw(show_yolo=False)

    # 📌 Mostrar imagen
    cv2.imshow('ViTPose + MediaPipe', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()

