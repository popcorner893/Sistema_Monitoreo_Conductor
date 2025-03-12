import cv2
import json
import os
import numpy as np
import pandas as pd

#Rutas de los archivos
video_path = r"D:\Cosas varias\Proyecto IA\dmd\gC\15\s5\gC_15_s5_2019-03-12T11;03;23+01;00_rgb_face.mp4"
json_path = r"D:\Cosas varias\Proyecto IA\dmd\gC\15\s5\gC_15_s5_2019-03-12T11;03;23+01;00_rgb_ann_drowsiness.json"
output_dir = r"D:\Cosas varias\Proyecto IA\Output\frames_clasificados_15"
all_frames_dir = os.path.join(output_dir, "all_frames")  # Carpeta para todos los frames

# Obtener el número real de frames en el video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"📌 El video tiene {total_frames} frames.")

# Cargar anotaciones JSON en forma OpenLabel
with open(json_path, "r") as f:
    annotations = json.load(f)

# Extraer categorías de acciones por cada frame
action_categories = set()
actions_data = annotations["openlabel"]["actions"]

# Obtener los nombres únicos de las categorías de acciones (ej. "eyes_state", "blinks", "yawning")
for action_info in actions_data.values():
    action_name = action_info["type"]
    category = action_name.split('/')[0]  # Tomar solo la primera parte antes del "/"
    action_categories.add(category)

action_categories = sorted(list(action_categories))  # Ordenar categorías para consistencia

# Inicializar diccionario `frames_actions`. A cada frame le corresponden categorías
frames_actions = {frame: {category: np.nan for category in action_categories} for frame in range(total_frames)}

# Llenar `frames_actions` con las acciones correspondientes a cada frame
for action_info in actions_data.values():
    action_name = action_info["type"]
    category = action_name.split('/')[0]  # Obtener la categoría (ej. "eyes_state"). En OpenLabel, el formato es categoria/acción
    
    for interval in action_info.get("frame_intervals", []):
        start_frame, end_frame = int(interval["frame_start"]), int(interval["frame_end"])
        
        # Evitar que los frames anotados sobrepasen el total del video
        if start_frame >= total_frames:
            continue
        if end_frame >= total_frames:
            end_frame = total_frames - 1
        
        # Asignar la acción al frame en su categoría correspondiente
        for frame in range(start_frame, end_frame + 1):
            frames_actions[frame][category] = action_name  # Guardar la acción específica

# Crear carpeta para todos los frames
os.makedirs(all_frames_dir, exist_ok=True)

# Volver a abrir el video para extraer y guardar frames en `all_frames/`, uno por uno y en orden
cap = cv2.VideoCapture(video_path)
frame_idx = 0
frames_saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= total_frames:
        break  # Fin del video

    # Guardar cada frame en "all_frames"
    frame_filename = os.path.join(all_frames_dir, f"frame_{frame_idx+1}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_idx += 1
    frames_saved += 1

cap.release()
cv2.destroyAllWindows()

print(f"\n Se guardaron {frames_saved} frames en 'all_frames/'.")

#  Convertir `frames_actions` en un DataFrame de Pandas
df_frames_actions = pd.DataFrame.from_dict(frames_actions, orient="index")
df_frames_actions["occlusion"] = df_frames_actions.get("occlusion", np.nan)


# Mostrar el DataFrame resultante
print("\n📌 DataFrame con anotaciones de acciones por frame:")
df_frames_actions.head()

# Guardar el DataFrame como CSV
csv_path = os.path.join(output_dir, "frames_actions.csv")
df_frames_actions.to_csv(csv_path, index=True)
print(f"\n✅ DataFrame guardado en: {csv_path}")