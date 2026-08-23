"""Recupera el fold1_v3 (self copy-paste) que se borro por error -- reproduce exactamente el
mismo fold (semilla identica) que ya dio F1=0.6037 en kfold_resultados_parciales_v3.json,
solo que ahora si guardando los pesos para poder usarlos en el ensemble."""

import sys
sys.path.insert(0, r"C:\Users\jchag\Documents\TESIS")
from kfold_cv_v3_selfcp import (
    RAIZ, SEMILLA, N_FOLDS, EPOCAS, IMGSZ, BATCH, PACIENCIA, IOU_EVAL,
    cargar_muestras, construir_dataset_yolo, sobremuestrear_propio, generar_selfcp_fold, evaluar,
    DIR_PROPIO_IMG, DIR_PROPIO_ANN, DIR_MD121_IMG, DIR_MD121_ANN,
)
import random
from ultralytics import YOLO

print("--- Reconstruyendo split (identico, misma semilla) ---")
muestras_publico = cargar_muestras(DIR_MD121_IMG, DIR_MD121_ANN, "md121")
for m in muestras_publico:
    m["grupo"] = "publico"
muestras_propio = cargar_muestras(DIR_PROPIO_IMG, DIR_PROPIO_ANN, "propio")
for m in muestras_propio:
    m["grupo"] = "propio"

rng_pub = random.Random(SEMILLA)
copia_pub = muestras_publico[:]
rng_pub.shuffle(copia_pub)
n = len(copia_pub)
corte_tr = int(n * 0.70)
corte_val = corte_tr + int(n * 0.15)
pub_train, pub_val, pub_test = copia_pub[:corte_tr], copia_pub[corte_tr:corte_val], copia_pub[corte_val:]

rng_folds = random.Random(SEMILLA)
copia_propio = muestras_propio[:]
rng_folds.shuffle(copia_propio)
tam_fold = len(copia_propio) // N_FOLDS
folds = [copia_propio[i * tam_fold:(i + 1) * tam_fold] for i in range(N_FOLDS)]

fold_idx = 0
nombre_fold = "fold1"
propio_test = folds[fold_idx]
resto = [img for j, f in enumerate(folds) if j != fold_idx for img in f]
rng_val = random.Random(SEMILLA + fold_idx)
rng_val.shuffle(resto)
n_val = max(1, round(len(resto) * 0.15))
propio_val = resto[:n_val]
propio_train = resto[n_val:]
print(f"Propio -> train={len(propio_train)} val={len(propio_val)} test={len(propio_test)}")

dir_cp = rf"{RAIZ}\Propio_selfcp_{nombre_fold}_v3"
print(f"--- Generando self copy-paste para {nombre_fold} ---")
muestras_cp = generar_selfcp_fold(propio_train, dir_cp, SEMILLA + fold_idx)
print(f"{len(muestras_cp)} imagenes sinteticas generadas")

train = pub_train + propio_train + muestras_cp
val = pub_val + propio_val
test = pub_test + propio_test
train, factor = sobremuestrear_propio(train)
print(f"Sobremuestreo factor x{factor}, train total = {len(train)} imagenes")

dir_dataset = rf"{RAIZ}\yolo_dataset_{nombre_fold}_v3"
ruta_yaml = construir_dataset_yolo(train, val, test, dir_dataset)

ruta_yaml_test_propio = rf"{RAIZ}\yolo_dataset_{nombre_fold}_test_propio_v2\data.yaml"

print(f"\n--- Entrenando {nombre_fold} (self copy-paste, patience={PACIENCIA}) ---")
modelo = YOLO("yolov8n.pt")
modelo.train(
    data=ruta_yaml, epochs=EPOCAS, imgsz=IMGSZ, batch=BATCH, patience=PACIENCIA,
    device=0, workers=0, degrees=15.0, flipud=0.5,
    project=rf"{RAIZ}\yolo_runs", name=f"whitefly_yolov8n_{nombre_fold}_v3",
)

mejor_pt = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_{nombre_fold}_v3\weights\best.pt"
resultado = evaluar(mejor_pt, ruta_yaml_test_propio, f"whitefly_{nombre_fold}_v3_recuperado_eval")
print(f"\n{nombre_fold}_v3 recuperado: P={resultado['P']:.4f} R={resultado['R']:.4f} F1={resultado['F1']:.4f}")
print("Para comparar (resultado original antes de borrar por error): F1=0.6037")
