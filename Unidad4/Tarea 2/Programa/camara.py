import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224

# Cargar modelo entrenado
model = tf.keras.models.load_model('modelo_plantas_cnn.h5')

# Etiquetas con índice explícito
class_names = [
    'African violet', 'Aglaonema', 'Anthurium', 'Areca palm', 'Bamboo plant',
    'Basil plant', 'Begonia', 'Boston fern', 'Caladium', 'Calathea',
    'Camellia', 'Chinese evergreen', 'Chrysanthemum', 'Coleus', 'Croton plant',
    'Curry leaf plant', 'Dahlia', 'Daisy plant', 'Dandelion', 'Dracaena',
    'Ficus plant', 'Geranium', 'Gladiolus', 'Heliconia', 'Hibiscus',
    'Ivy plant', 'Jade plant', 'Jasmine', 'Kalanchoe', 'Lavender',
    'Lotus', 'Marigold', 'Mint plant', 'Money plant', 'Oleander',
    'Orchid', 'Peace lily', 'Peony', 'Petunia', 'Plumeria',
    'Pothos', 'Rose Plant', 'Rosemary', 'Sage plant', 'Snake plant',
    'Snapdragon', 'Spider plant', 'Thyme', 'Tulip plant', 'Verbena',
    'Zinnia plant', 'ZZ plant'
]

# Abrir cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_norm = img / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    preds = model.predict(img_batch, verbose=0)
    confidence = np.max(preds)
    class_index = np.argmax(preds)

    if confidence < 0.5:
        pred_label = "No reconocido"
        text = f"{pred_label} ({confidence*100:.1f}%)"
    else:
        pred_label = class_names[class_index]
        text = f"[{class_index}] {pred_label} ({confidence*100:.1f}%)"

    org = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color_text = (255, 255, 255)  # Blanco
    color_box = (255, 0, 0)       # Azul en BGR
    thickness = 2

    # Obtener tamaño del texto para el recuadro
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Dibujar fondo azul (recuadro)
    cv2.rectangle(frame, (org[0] - 5, org[1] - text_h - 10), (org[0] + text_w + 5, org[1] + 5), color_box, cv2.FILLED)

    # Poner texto encima del recuadro
    cv2.putText(frame, text, org, font, font_scale, color_text, thickness)

    cv2.imshow("Clasificador de Plantas", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
