"""Etapa 19: reentrenamiento de combinado-6 con las etiquetas propias corregidas (2026-08-21).
Se encontraron ~11 imagenes propias masivamente sub-etiquetadas (ej. 2079: 1->116 cajas,
2081: 1->58, 2080: 1->96) que nunca pasaron por la auditoria de la Etapa 15 -- el 5-fold CV
(Etapa 18) las destapo. Se corrigieron en Roboflow y se re-exportaron.

Receta IDENTICA a combinado-6 (yolo_entrenamiento.py, MODO_DATASET="combinado"), la UNICA
diferencia es que Propio\\Tesis.voc\\annotations y Propio_copypaste ya reflejan los datos
corregidos. Carpetas de salida separadas para no pisar el combinado-6 original (se conserva
como referencia historica)."""

import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

FUENTES = [
    {
        "nombre": "md121",
        "grupo": "publico",
        "dir_imagenes": r"C:\Users\jchag\Documents\TESIS\md121\images",
        "dir_anotaciones": r"C:\Users\jchag\Documents\TESIS\md121\annotations",
    },
    {
        "nombre": "propio",
        "grupo": "propio",
        "dir_imagenes": r"C:\Users\jchag\Documents\TESIS\Propio\Tesis.voc\imagess",
        "dir_anotaciones": r"C:\Users\jchag\Documents\TESIS\Propio\Tesis.voc\annotations",
    },
]

INCLUIR_COPY_PASTE = True
DIR_COPY_PASTE_IMAGENES = r"C:\Users\jchag\Documents\TESIS\Propio_copypaste\images"
DIR_COPY_PASTE_ANOTACIONES = r"C:\Users\jchag\Documents\TESIS\Propio_copypaste\annotations"

CLASES = ["WF"]
DIR_DATASET_YOLO = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_combinado6_v2"
MAX_DIM_DATASET = 1920
PROPORCION_VAL = 0.15
PROPORCION_TEST = 0.15
SEMILLA = 42
SOBREMUESTREAR_PROPIO = True

MODELO_BASE = "yolov8n.pt"
EPOCAS = 100
IMGSZ = 1280
BATCH = 4
PACIENCIA = 20
NOMBRE_RUN = "whitefly_yolov8n_combinado6_v2"


def parsear_voc(ruta_xml, clases_validas):
    cajas = []
    try:
        root = ET.parse(ruta_xml).getroot()
        for obj in root.findall("object"):
            nombre_clase = obj.find("name").text
            if nombre_clase not in clases_validas:
                continue
            b = obj.find("bndbox")
            cajas.append([nombre_clase, float(b.find("xmin").text), float(b.find("ymin").text),
                          float(b.find("xmax").text), float(b.find("ymax").text)])
    except Exception as e:
        print(f"Error leyendo {ruta_xml}: {e}")
    return cajas


def caja_a_yolo(clase_idx, xmin, ymin, xmax, ymax, img_w, img_h):
    xc = ((xmin + xmax) / 2.0) / img_w
    yc = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"{clase_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def recolectar_muestras():
    muestras_por_fuente = {}
    for fuente in FUENTES:
        muestras = []
        for ruta_xml in glob.glob(os.path.join(fuente["dir_anotaciones"], "*.xml")):
            nombre_base = os.path.splitext(os.path.basename(ruta_xml))[0]
            ruta_img = os.path.join(fuente["dir_imagenes"], nombre_base + ".jpg")
            if not os.path.exists(ruta_img):
                continue
            muestras.append({
                "nombre_salida": f"{fuente['nombre']}_{nombre_base}",
                "ruta_imagen": ruta_img,
                "cajas": parsear_voc(ruta_xml, CLASES),
                "grupo": fuente["grupo"],
            })
        muestras_por_fuente[fuente["nombre"]] = muestras
        print(f"[{fuente['nombre']}] {len(muestras)} imagenes con anotacion emparejada.")
    return muestras_por_fuente


def cargar_muestras_copy_paste():
    muestras = []
    for ruta_xml in glob.glob(os.path.join(DIR_COPY_PASTE_ANOTACIONES, "*.xml")):
        nombre_base = os.path.splitext(os.path.basename(ruta_xml))[0]
        ruta_img = os.path.join(DIR_COPY_PASTE_IMAGENES, nombre_base + ".jpg")
        if not os.path.exists(ruta_img):
            continue
        muestras.append({
            "nombre_salida": f"cp_{nombre_base}",
            "ruta_imagen": ruta_img,
            "cajas": parsear_voc(ruta_xml, CLASES),
            "grupo": "propio",
        })
    return muestras


def dividir_train_val_test(muestras_por_fuente, proporcion_val, proporcion_test, semilla):
    train, val, test = [], [], []
    rng = random.Random(semilla)
    for muestras in muestras_por_fuente.values():
        copia = muestras[:]
        rng.shuffle(copia)
        n = len(copia)
        corte_tr = int(n * (1 - proporcion_val - proporcion_test))
        corte_val = corte_tr + int(n * proporcion_val)
        train.extend(copia[:corte_tr])
        val.extend(copia[corte_tr:corte_val])
        test.extend(copia[corte_val:])
    return train, val, test


def sobremuestrear_propio(train):
    propio = [m for m in train if m["grupo"] == "propio"]
    resto = [m for m in train if m["grupo"] != "propio"]
    if not propio or not resto:
        return train, 1
    factor = max(1, round(len(resto) / len(propio)))
    if factor <= 1:
        return train, 1
    rep = []
    for n in range(factor):
        for m in propio:
            c = dict(m)
            c["nombre_salida"] = f"{m['nombre_salida']}_rep{n}"
            rep.append(c)
    return resto + rep, factor


def construir_dataset_yolo(train, val, test, dir_salida):
    for sub in ("images", "labels"):
        p = os.path.join(dir_salida, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for split, muestras in [("train", train), ("val", val), ("test", test)]:
        os.makedirs(os.path.join(dir_salida, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dir_salida, "labels", split), exist_ok=True)
        for m in muestras:
            nombre = m["nombre_salida"]
            ruta_img_dst = os.path.join(dir_salida, "images", split, nombre + ".jpg")
            ruta_lbl_dst = os.path.join(dir_salida, "labels", split, nombre + ".txt")
            with Image.open(m["ruta_imagen"]) as img:
                img = img.convert("RGB")
                w, h = img.size
                if max(w, h) > MAX_DIM_DATASET:
                    escala = MAX_DIM_DATASET / max(w, h)
                    img = img.resize((int(w * escala), int(h * escala)), Image.LANCZOS)
                img.save(ruta_img_dst, "JPEG", quality=95)
            lineas = [caja_a_yolo(CLASES.index(c[0]), c[1], c[2], c[3], c[4], w, h) for c in m["cajas"]]
            with open(ruta_lbl_dst, "w") as f:
                f.write("\n".join(lineas))
    ruta_yaml = os.path.join(dir_salida, "data.yaml")
    with open(ruta_yaml, "w") as f:
        f.write(f"path: {dir_salida}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n  0: WF\n")
    return ruta_yaml


def contar_cajas(muestras):
    return sum(len(m["cajas"]) for m in muestras)


if __name__ == "__main__":
    print("--- Recolectando muestras (imagen + XML emparejados) ---")
    muestras_por_fuente = recolectar_muestras()

    print("\n--- Dividiendo train/val/test 70/15/15 (estratificado por fuente) ---")
    train, val, test = dividir_train_val_test(muestras_por_fuente, PROPORCION_VAL, PROPORCION_TEST, SEMILLA)
    print(f"Train: {len(train)} imagenes, {contar_cajas(train)} cajas WF")
    print(f"Val:   {len(val)} imagenes, {contar_cajas(val)} cajas WF")
    print(f"Test:  {len(test)} imagenes, {contar_cajas(test)} cajas WF")

    if INCLUIR_COPY_PASTE:
        muestras_cp = cargar_muestras_copy_paste()
        train.extend(muestras_cp)
        print(f"\n+{len(muestras_cp)} imagenes sinteticas, +{contar_cajas(muestras_cp)} cajas WF")
        print(f"Train total tras copy-paste: {len(train)} imagenes, {contar_cajas(train)} cajas WF")

    if SOBREMUESTREAR_PROPIO:
        n_antes = len([m for m in train if m["grupo"] == "propio"])
        train, factor = sobremuestrear_propio(train)
        n_despues = len([m for m in train if m["grupo"] == "propio"])
        print(f"\nSobremuestreo factor x{factor}: propio {n_antes} -> {n_despues}")
        print(f"Train total tras sobremuestreo: {len(train)} imagenes, {contar_cajas(train)} cajas WF")

    print(f"\n--- Construyendo dataset YOLO en: {DIR_DATASET_YOLO} ---")
    ruta_yaml = construir_dataset_yolo(train, val, test, DIR_DATASET_YOLO)
    print(f"data.yaml generado en: {ruta_yaml}")

    from ultralytics import YOLO
    print(f"\n--- Entrenando {MODELO_BASE} sobre {ruta_yaml} ---")
    modelo = YOLO(MODELO_BASE)
    modelo.train(
        data=ruta_yaml, epochs=EPOCAS, imgsz=IMGSZ, batch=BATCH, patience=PACIENCIA,
        device=0, workers=0, degrees=15.0, flipud=0.5,
        project=r"C:\Users\jchag\Documents\TESIS\yolo_runs", name=NOMBRE_RUN,
    )
