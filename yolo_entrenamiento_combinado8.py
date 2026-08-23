"""
Copia de yolo_entrenamiento.py para la Etapa 12 (whitefly_yolov8n_combinado-8): "self copy-paste"
-- moscas recortadas del propio dataset (no de md121) pegadas sobre fondos propios reales, misma
densidad que combinado-6 (Etapa 6) para aislar una sola variable: origen del recorte.
Ver copy_paste_augmentation_v3.py.
"""

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

MODO_DATASET = "combinado8"
FUENTES_ACTIVAS = FUENTES

INCLUIR_COPY_PASTE = True
DIR_COPY_PASTE_IMAGENES = r"C:\Users\jchag\Documents\TESIS\Propio_copypaste_v3\images"
DIR_COPY_PASTE_ANOTACIONES = r"C:\Users\jchag\Documents\TESIS\Propio_copypaste_v3\annotations"

CLASES = ["WF"]
DIR_DATASET_YOLO = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_combinado8"
MAX_DIM_DATASET = 1920
PROPORCION_VAL = 0.15
PROPORCION_TEST = 0.15
SEMILLA = 42

SOBREMUESTREAR_PROPIO = True

CONVERTIR_DATASET = True
ENTRENAR = True

MODELO_BASE = "yolov8n.pt"
EPOCAS = 100
IMGSZ = 1280
BATCH = 4
PACIENCIA = 20


def parsear_voc(ruta_xml, clases_validas):
    cajas = []
    try:
        root = ET.parse(ruta_xml).getroot()
        for obj in root.findall("object"):
            nombre_clase = obj.find("name").text
            if nombre_clase not in clases_validas:
                continue
            bndbox = obj.find("bndbox")
            cajas.append([
                nombre_clase,
                float(bndbox.find("xmin").text),
                float(bndbox.find("ymin").text),
                float(bndbox.find("xmax").text),
                float(bndbox.find("ymax").text),
            ])
    except Exception as e:
        print(f"Error leyendo {ruta_xml}: {e}")
    return cajas


def caja_a_yolo(clase_idx, xmin, ymin, xmax, ymax, img_w, img_h):
    x_centro = ((xmin + xmax) / 2.0) / img_w
    y_centro = ((ymin + ymax) / 2.0) / img_h
    ancho = (xmax - xmin) / img_w
    alto = (ymax - ymin) / img_h
    return f"{clase_idx} {x_centro:.6f} {y_centro:.6f} {ancho:.6f} {alto:.6f}"


def recolectar_muestras():
    muestras_por_fuente = {}
    for fuente in FUENTES_ACTIVAS:
        muestras = []
        rutas_xml = glob.glob(os.path.join(fuente["dir_anotaciones"], "*.xml"))
        for ruta_xml in rutas_xml:
            nombre_base = os.path.splitext(os.path.basename(ruta_xml))[0]
            ruta_img = os.path.join(fuente["dir_imagenes"], nombre_base + ".jpg")
            if not os.path.exists(ruta_img):
                continue
            cajas = parsear_voc(ruta_xml, CLASES)
            muestras.append({
                "nombre_salida": f"{fuente['nombre']}_{nombre_base}",
                "ruta_imagen": ruta_img,
                "cajas": cajas,
                "grupo": fuente["grupo"],
            })
        muestras_por_fuente[fuente["nombre"]] = muestras
        print(f"[{fuente['nombre']}] {len(muestras)} imágenes con anotación emparejada.")
    return muestras_por_fuente


def cargar_muestras_copy_paste():
    muestras = []
    rutas_xml = glob.glob(os.path.join(DIR_COPY_PASTE_ANOTACIONES, "*.xml"))
    for ruta_xml in rutas_xml:
        nombre_base = os.path.splitext(os.path.basename(ruta_xml))[0]
        ruta_img = os.path.join(DIR_COPY_PASTE_IMAGENES, nombre_base + ".jpg")
        if not os.path.exists(ruta_img):
            continue
        cajas = parsear_voc(ruta_xml, CLASES)
        muestras.append({
            "nombre_salida": f"cp_{nombre_base}",
            "ruta_imagen": ruta_img,
            "cajas": cajas,
            "grupo": "propio",
        })
    return muestras


def dividir_train_val_test(muestras_por_fuente, proporcion_val, proporcion_test, semilla):
    train, val, test = [], [], []
    rng = random.Random(semilla)
    for nombre_fuente, muestras in muestras_por_fuente.items():
        copia = muestras[:]
        rng.shuffle(copia)
        n = len(copia)
        corte_train = int(n * (1 - proporcion_val - proporcion_test))
        corte_val = corte_train + int(n * proporcion_val)
        train.extend(copia[:corte_train])
        val.extend(copia[corte_train:corte_val])
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

    propio_sobremuestreado = []
    for copia_n in range(factor):
        for m in propio:
            copia = dict(m)
            copia["nombre_salida"] = f"{m['nombre_salida']}_rep{copia_n}"
            propio_sobremuestreado.append(copia)

    return resto + propio_sobremuestreado, factor


def construir_dataset_yolo(train, val, test, dir_salida):
    for sub in ("images", "labels"):
        ruta_sub = os.path.join(dir_salida, sub)
        if os.path.isdir(ruta_sub):
            shutil.rmtree(ruta_sub)

    for split, muestras in [("train", train), ("val", val), ("test", test)]:
        os.makedirs(os.path.join(dir_salida, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dir_salida, "labels", split), exist_ok=True)

        for muestra in muestras:
            nombre = muestra["nombre_salida"]
            ruta_img_destino = os.path.join(dir_salida, "images", split, nombre + ".jpg")
            ruta_label_destino = os.path.join(dir_salida, "labels", split, nombre + ".txt")

            with Image.open(muestra["ruta_imagen"]) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size
                if max(img_w, img_h) > MAX_DIM_DATASET:
                    escala = MAX_DIM_DATASET / max(img_w, img_h)
                    img = img.resize((int(img_w * escala), int(img_h * escala)), Image.LANCZOS)
                img.save(ruta_img_destino, "JPEG", quality=95)

            lineas = []
            for clase, xmin, ymin, xmax, ymax in muestra["cajas"]:
                clase_idx = CLASES.index(clase)
                lineas.append(caja_a_yolo(clase_idx, xmin, ymin, xmax, ymax, img_w, img_h))

            with open(ruta_label_destino, "w") as f:
                f.write("\n".join(lineas))

    ruta_yaml = os.path.join(dir_salida, "data.yaml")
    with open(ruta_yaml, "w") as f:
        f.write(f"path: {dir_salida}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n")
        for i, clase in enumerate(CLASES):
            f.write(f"  {i}: {clase}\n")

    return ruta_yaml


def contar_cajas(muestras):
    return sum(len(m["cajas"]) for m in muestras)


if __name__ == "__main__":
    ruta_data_yaml = os.path.join(DIR_DATASET_YOLO, "data.yaml")
    print(f"=== MODO_DATASET = {MODO_DATASET} (combinado8: self copy-paste, recortes propios) ===")

    if CONVERTIR_DATASET:
        print("--- Recolectando muestras (imagen + XML emparejados) ---")
        muestras_por_fuente = recolectar_muestras()

        print("\n--- Dividiendo train/val/test 70/15/15 (estratificado por fuente) ---")
        train, val, test = dividir_train_val_test(muestras_por_fuente, PROPORCION_VAL, PROPORCION_TEST, SEMILLA)
        print(f"Train: {len(train)} imágenes, {contar_cajas(train)} cajas WF")
        print(f"Val:   {len(val)} imágenes, {contar_cajas(val)} cajas WF")
        print(f"Test:  {len(test)} imágenes, {contar_cajas(test)} cajas WF")

        if INCLUIR_COPY_PASTE:
            muestras_cp = cargar_muestras_copy_paste()
            train.extend(muestras_cp)
            print(f"\n--- Self copy-paste (recortes propios) agregado a train ---")
            print(f"+{len(muestras_cp)} imágenes sintéticas, +{contar_cajas(muestras_cp)} cajas WF")
            print(f"Train total tras copy-paste: {len(train)} imágenes, {contar_cajas(train)} cajas WF")

        if SOBREMUESTREAR_PROPIO:
            n_propio_antes = len([m for m in train if m["grupo"] == "propio"])
            train, factor = sobremuestrear_propio(train)
            n_propio_despues = len([m for m in train if m["grupo"] == "propio"])
            print(f"\n--- Sobremuestreo de 'propio' en train (factor x{factor}) ---")
            print(f"Propio en train: {n_propio_antes} -> {n_propio_despues} (resto sin tocar)")
            print(f"Train total tras sobremuestreo: {len(train)} imágenes, {contar_cajas(train)} cajas WF")

        print(f"\n--- Construyendo dataset YOLO en: {DIR_DATASET_YOLO} ---")
        ruta_data_yaml = construir_dataset_yolo(train, val, test, DIR_DATASET_YOLO)
        print(f"data.yaml generado en: {ruta_data_yaml}")

    if ENTRENAR:
        from ultralytics import YOLO

        print(f"\n--- Entrenando {MODELO_BASE} sobre {ruta_data_yaml} ---")
        modelo = YOLO(MODELO_BASE)
        modelo.train(
            data=ruta_data_yaml,
            epochs=EPOCAS,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PACIENCIA,
            device=0,
            workers=0,
            degrees=15.0,
            flipud=0.5,
            project=r"C:\Users\jchag\Documents\TESIS\yolo_runs",
            name="whitefly_yolov8n_combinado-8",
        )
