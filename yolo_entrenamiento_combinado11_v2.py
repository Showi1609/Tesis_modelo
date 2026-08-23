"""Etapa 24: "negativos graduales". Retoma la idea descartada en la Etapa 17 (combinado-10: 24
fotos propias sin mosca blanca como negativos duros, con etiquetas VIEJAS -- no probado con las
corregidas) pero cambiando COMO entran a train.

Hipotesis (motivada por el hallazgo de que 94/284 imagenes publicas ya tienen 0 cajas WF de forma
NATURAL, distribuidas sin problema): lo que daño a combinado-10 no fueron los negativos en si, sino
la CONCENTRACION -- en combinado-10 los 24 negativos entraban a train ANTES del sobremuestreo de
"propio", asi que se multiplicaban por el mismo factor (~x4) que el resto de imagenes propias,
terminando como ~96 copias casi identicas de "no hay nada aqui" en un train de pocos cientos de
imagenes. Aca entran DESPUES del sobremuestreo, una sola vez cada una (24 imagenes reales, sin
duplicar) -- diluidas en el train igual que los negativos naturales de publico.

Receta identica a combinado-6_v2 (Etapa 19, etiquetas propias corregidas) en todo lo demas."""

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

INCLUIR_NEGATIVOS = True
DIR_NEGATIVOS_IMAGENES = r"C:\Users\jchag\Documents\TESIS\Propio_negativos\images"
DIR_NEGATIVOS_ANOTACIONES = r"C:\Users\jchag\Documents\TESIS\Propio_negativos\annotations"

CLASES = ["WF"]
DIR_DATASET_YOLO = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_combinado11_v2"
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
NOMBRE_RUN = "whitefly_yolov8n_combinado11_v2"


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


def cargar_muestras_negativos():
    """24 fotos propias reales sin mosca blanca (trips, pulgones, etc.), 0 cajas -- nunca a val/test."""
    muestras = []
    for ruta_xml in glob.glob(os.path.join(DIR_NEGATIVOS_ANOTACIONES, "*.xml")):
        nombre_base = os.path.splitext(os.path.basename(ruta_xml))[0]
        ruta_img = os.path.join(DIR_NEGATIVOS_IMAGENES, nombre_base + ".jpg")
        if not os.path.exists(ruta_img):
            continue
        muestras.append({
            "nombre_salida": nombre_base,
            "ruta_imagen": ruta_img,
            "cajas": parsear_voc(ruta_xml, CLASES),  # siempre [] -- XML vacios
            "grupo": "negativo",  # OJO: NO "propio" -- no debe entrar al sobremuestreo
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

    if INCLUIR_NEGATIVOS:
        # A PROPOSITO despues del sobremuestreo: entran una sola vez cada una, sin duplicar,
        # diluidas en el train ya construido -- no reciben el factor x del resto de "propio".
        muestras_neg = cargar_muestras_negativos()
        train.extend(muestras_neg)
        pct = 100 * len(muestras_neg) / len(train)
        print(f"\n+{len(muestras_neg)} negativos duros (sin duplicar, {pct:.1f}% del train final), +0 cajas")
        print(f"Train total final: {len(train)} imagenes, {contar_cajas(train)} cajas WF")

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

    mejor_pt = rf"C:\Users\jchag\Documents\TESIS\yolo_runs\{NOMBRE_RUN}\weights\best.pt"
    modelo_eval = YOLO(mejor_pt)
    RAIZ = r"C:\Users\jchag\Documents\TESIS"
    for nombre_subset, data_yaml in [
        ("propio", rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"),
        ("publico", rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"),
    ]:
        metricas = modelo_eval.val(
            data=data_yaml, split="val", imgsz=IMGSZ, iou=0.45, device=0, workers=0, verbose=False,
            project=rf"{RAIZ}\yolo_runs", name=f"whitefly_combinado11_v2_{nombre_subset}",
        )
        p, r = metricas.box.mp, metricas.box.mr
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"\n=== combinado11_v2 (negativos graduales) -- {nombre_subset} ===")
        print(f"Precision: {p:.4f}  Recall: {r:.4f}  mAP50: {metricas.box.map50:.4f}  F1: {f1:.4f}")

    print("\nPara comparar: combinado-6_v2 (sin negativos): F1 propio=0.6705 F1 publico=0.8556")
