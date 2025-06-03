import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from joblib import load
from easy_ViTPose import VitInference
from PIL import Image
import os

# Configuración
MODEL_SIZE = 'b'
DATASET = 'wholebody'
model_path = r"D:\Cosas varias\Proyecto IA\vitpose-b-wholebody.pth"
yolo_path = r"D:\Cosas varias\Proyecto IA\yolov8s.pt"
output_video_path = r"D:\Miércoles varias\BackupIA\Video\VideoConPosturas.mp4"
temp_frames_dir = r"D:\Miércoles varias\BackupIA\Temp"
alert_icon_path = r"D:\Cosas varias\Proyecto IA\alert_icon.png"  # Ruta al ícono

# Configuración de alerta
ventana_n_frames = 90
w1, w2, w3 = 1.5, 1.2, 0.3
umbral_alerta = 60

# Cargar ícono de alerta
alert_icon = cv2.imread(alert_icon_path, cv2.IMREAD_UNCHANGED)
icon_size = 80
alert_icon = cv2.resize(alert_icon, (icon_size, icon_size))

# Cargar modelos entrenados
scaler_blinks = load(r"D:\Miércoles varias\modelos_fatiga-20250601T022515Z-1-001\modelos_fatiga\scaler_blinks.pkl")
scaler_eyes = load(r"D:\Miércoles varias\modelos_fatiga-20250601T022515Z-1-001\modelos_fatiga\scaler_eyes_state.pkl")
scaler_yawning = load(r"D:\Miércoles varias\modelos_fatiga-20250601T022515Z-1-001\modelos_fatiga\scaler_yawning.pkl")

pca_blinks = load(r"D:\Miércoles varias\modelos_fatiga-20250601T022515Z-1-001\modelos_fatiga\pca_blinks.pkl")
pca_eyes = load(r"D:\Miércoles varias\modelos_fatiga-20250601T022515Z-1-001\modelos_fatiga\pca_eyes_state.pkl")
pca_yawning = load(r"D:\Miércoles varias\modelos_fatiga-20250601T022515Z-1-001\modelos_fatiga\pca_yawning.pkl")

model_blinks = load(r"D:\Miércoles varias\modelos_fatiga-20250601T022515Z-1-001\modelos_fatiga\rf_model_blinks.pkl")
model_eyes = load(r"D:\Miércoles varias\modelos_fatiga-20250601T022515Z-1-001\modelos_fatiga\rf_model_eyes_state.pkl")
model_yawning = load(r"D:\Miércoles varias\modelos_fatiga-20250601T022515Z-1-001\modelos_fatiga\rf_model_yawning.pkl")

# Diccionarios de etiquetas
blinks_map_rev = {0: 'no blink', 1: 'blinking'}
eyes_state_map_rev = {0: 'undefined', 1: 'open', 2: 'closed', 3: 'opening', 4: 'closing', 5: 'undefined'}
yawning_map_rev = {0: 'no yawning', 1: 'Yawning w/o hand', 2: 'Yawning w/ hand'}

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)
model = VitInference(model_path, yolo_path, MODEL_SIZE, dataset=DATASET, yolo_size=320, is_video=False)

# Funciones auxiliares 
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    h, w = alpha_mask.shape
    alpha_inv = 1.0 - alpha_mask

    for c in range(3):
        img[y:y+h, x:x+w, c] = (
            alpha_mask * img_overlay[:, :, c] +
            alpha_inv * img[y:y+h, x:x+w, c]
        )

def get_face_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    return results.multi_face_landmarks if results.multi_face_landmarks else None

def get_landmarks_dataframe(face_landmarks, image_shape):
    if not face_landmarks:
        return None
    h, w, _ = image_shape
    landmarks_dict = {}
    for idx, landmark in enumerate(face_landmarks[0].landmark):
        landmarks_dict[f'lmk_{idx}_x'] = landmark.x
        landmarks_dict[f'lmk_{idx}_y'] = landmark.y
        landmarks_dict[f'lmk_{idx}_z'] = landmark.z
    return pd.DataFrame([landmarks_dict])

def get_vitpose_landmarks(img):
    LANDMARKS_DEPEND_ON_9 = [91, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110]
    LANDMARKS_DEPEND_ON_10 = [112, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131]
    frame_keypoints = model.inference(img)
    bboxes, ids, scores = model._tracker_res
    keypoints = frame_keypoints[0] if frame_keypoints else []
    bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bboxes[0] if len(bboxes) > 0 else (0, 0, img.shape[1], img.shape[0])

    def is_inside_bbox(x, y):
        return bbox_x_min <= x <= bbox_x_max and bbox_y_min <= y <= bbox_y_max

    h, w, _ = img.shape
    landmarks_dict = {}
    for idx in LANDMARKS_DEPEND_ON_9 + LANDMARKS_DEPEND_ON_10:
        if idx < len(keypoints):
            y, x, conf = keypoints[idx]
            inside = is_inside_bbox(x, y)
            if inside and conf >= 0.7:
                landmarks_dict[f'lmk_hand_{idx}_x'] = x / w
                landmarks_dict[f'lmk_hand_{idx}_y'] = y / h
            else:
                landmarks_dict[f'lmk_hand_{idx}_x'] = np.nan
                landmarks_dict[f'lmk_hand_{idx}_y'] = np.nan
    return pd.DataFrame([landmarks_dict])

# Procesamiento principal 
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    historial_blinks, historial_eyes, historial_yawn = [], [], []
    all_predictions = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face_landmarks = get_face_landmarks(frame)
        df_face = get_landmarks_dataframe(face_landmarks, frame.shape)
        df_body = get_vitpose_landmarks(frame)

        if df_face is not None:
            df = pd.concat([df_face, df_body], axis=1).fillna(0)

            X_blinks_pca = pca_blinks.transform(scaler_blinks.transform(df))
            X_eyes_pca = pca_eyes.transform(scaler_eyes.transform(df))
            X_yawning_pca = pca_yawning.transform(scaler_yawning.transform(df))

            pred_blink = model_blinks.predict(X_blinks_pca)[0]
            pred_eyes = model_eyes.predict(X_eyes_pca)[0]
            pred_yawn = model_yawning.predict(X_yawning_pca)[0]

            pred_dict = {
                "blinks": blinks_map_rev[pred_blink],
                "eyes_state": eyes_state_map_rev[pred_eyes],
                "yawning": yawning_map_rev[pred_yawn]
            }
            all_predictions.append(pred_dict)

            # Agregar a historial
            historial_blinks.append(pred_blink)
            historial_eyes.append(pred_eyes)
            historial_yawn.append(pred_yawn)

            if len(historial_blinks) > ventana_n_frames:
                historial_blinks.pop(0)
                historial_eyes.pop(0)
                historial_yawn.pop(0)

            # Calcular suma de pesos
            suma_peso = (
                w1 * sum(1 for y in historial_yawn if y in [1, 2]) +
                w2 * sum(1 for e in historial_eyes if e in [2]) +
                w3 * sum(1 for b in historial_blinks if b == 1)
            )

            # Mostrar texto
            cv2.putText(frame, f"Blink: {pred_dict['blinks']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Eyes: {pred_dict['eyes_state']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Yawn: {pred_dict['yawning']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if suma_peso > umbral_alerta:
                cv2.putText(frame, "--ALERTA DE ACCION PELIGROSA --", (int(width*0.5)-250, int(height*0.2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 10)

                # Superponer ícono
                x_offset = int(width*0.5) - icon_size - 10
                y_offset = int(height*0.2) - int(icon_size/2)
                if alert_icon.shape[2] == 4:
                    overlay_image_alpha(frame, alert_icon[:, :, :3], (x_offset, y_offset), alert_icon[:, :, 3] / 255.0)
                else:
                    frame[y_offset:y_offset+icon_size, x_offset:x_offset+icon_size] = alert_icon

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("✅ Video procesado:", output_path)

    df_pred = pd.DataFrame(all_predictions)
    df_pred.to_csv(output_path.replace(".mp4", "_predicciones.csv"), index=False)
    print("📄 CSV guardado.")

# Ejecutar
video_path = r"D:\Cosas varias\Proyecto IA\dmd\gA\1\s5\gA_1_s5_2019-03-14T14;26;17+01;00_rgb_face.mp4"
output_video_path = r"D:\Miércoles varias\BackupIA\OutputVideo\salida_con_predicciones_alerta.mp4"
process_video(video_path, output_video_path)
