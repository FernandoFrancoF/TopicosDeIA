import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
DATASET_DIR = "dataset_plantas/raw"
DESTINO = "dataset_plantas/procesado"
AUMENTAR = True

def validar_imagen(path):
    try:
        img = cv2.imread(path)
        return img is not None and img.shape[0] >= IMG_SIZE and img.shape[1] >= IMG_SIZE
    except:
        return False

def augmentar(img):
    augmented = []
    # Volteo horizontal
    augmented.append(cv2.flip(img, 1))
    # Rotación 15°
    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), 15, 1)
    augmented.append(cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE)))
    # Ruido
    ruido = np.random.normal(0, 10, img.shape).astype(np.uint8)
    augmented.append(cv2.add(img, ruido))
    return augmented

def procesar_img(path, output_dir, nombre_base, aplicar_aug):
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(f"{output_dir}/{nombre_base}.jpg", img)
        if aplicar_aug:
            for i, aug in enumerate(augmentar(img)):
                cv2.imwrite(f"{output_dir}/{nombre_base}_aug{i}.jpg", aug)
        return True
    except:
        return False

def procesar_dataset():
    clases = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

    for split in ["entrenamiento", "validacion"]:
        for clase in clases:
            os.makedirs(os.path.join(DESTINO, split, clase), exist_ok=True)

    for clase in clases:
        ruta = os.path.join(DATASET_DIR, clase)
        imagenes = [f for f in os.listdir(ruta) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        imagenes_validas = [f for f in imagenes if validar_imagen(os.path.join(ruta, f))]

        train_imgs, val_imgs = train_test_split(imagenes_validas, test_size=0.2, random_state=42)

        # Procesar entrenamiento
        for idx, nombre in enumerate(tqdm(train_imgs, desc=f"[{clase}] entrenamiento")):
            ruta_origen = os.path.join(ruta, nombre)
            destino = os.path.join(DESTINO, "entrenamiento", clase)
            procesar_img(ruta_origen, destino, f"{clase}_{idx}", aplicar_aug=AUMENTAR)

        # Procesar validación
        for idx, nombre in enumerate(tqdm(val_imgs, desc=f"[{clase}] validación")):
            ruta_origen = os.path.join(ruta, nombre)
            destino = os.path.join(DESTINO, "validacion", clase)
            procesar_img(ruta_origen, destino, f"{clase}_{idx}", aplicar_aug=False)

    print("Dataset procesado exitosamente.")

if __name__ == "__main__":
    procesar_dataset()
