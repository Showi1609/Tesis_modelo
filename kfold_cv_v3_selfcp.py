"""
Etapa 22: 5-fold CV para la receta de combinado-8_v2 (self copy-paste, 100% recortes propios),
para completar el par que falta y poder: (a) validar F1 propio robusto de v8 por si mismo, y
(b) ensemblear cada fold con su par ya entrenado de combinado-6 (recortes publicos, Etapa 19b)
via WBF, todo con el mismo split de test exacto (misma semilla=42, mismos 5 folds de 12 imagenes).

Reutiliza los datasets de test ya construidos (yolo_dataset_fold{N}_test_propio_v2) -- el split
propio es identico al del k-fold v2 anterior, no hace falta reconstruirlo.

Aprendizaje de la Etapa 21: patience=40 en vez de 20, para no repetir el corte prematuro que le
costo ~6.8 pts de F1 a fold1 la primera vez.
"""

import glob
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DIR_PROPIO_IMG = rf"{RAIZ}\Propio\Tesis.voc\imagess"
DIR_PROPIO_ANN = rf"{RAIZ}\Propio\Tesis.voc\annotations"
DIR_MD121_IMG = rf"{RAIZ}\md121\images"
DIR_MD121_ANN = rf"{RAIZ}\md121\annotations"

CLASES = ["WF"]
N_FOLDS = 5
SEMILLA = 42
MAX_DIM_DATASET = 1920
IMGSZ = 1280
EPOCAS = 100
BATCH = 4
PACIENCIA = 40  # Etapa 21: subido de 20 a 40
IOU_EVAL = 0.45

# self copy-paste: misma config que combinado-8_v2 (Etapa 12 v2)
VARIANTES_POR_IMAGEN = 3
MOSCAS_POR_VARIANTE_MIN = 3
MOSCAS_POR_VARIANTE_MAX = 8
MAX_RECORTES_POR_IMAGEN_PROPIO = 18
PADDING_RECORTE = 0.15
MAX_INTENTOS_UBICACION = 30
SOLAPAMIENTO_MAX = 0.05

import cv2
from PIL import ImageDraw, ImageFilter


def parsear_voc(ruta_xml):
    cajas = []
    try:
        root = ET.parse(ruta_xml).getroot()
        for obj in root.findall("object"):
            nombre_clase = obj.find("name").text
            if nombre_clase not in CLASES:
                continue
            b = obj.find("bndbox")
            cajas.append(["WF", float(b.find("xmin").text), float(b.find("ymin").text),
                          float(b.find("xmax").text), float(b.find("ymax").text)])
    except Exception as e:
        print(f"Error leyendo {ruta_xml}: {e}")
    return cajas


def caja_a_yolo(clase_idx, xmin, ymin, xmax, ymax, w, h):
    xc = ((xmin + xmax) / 2.0) / w
    yc = ((ymin + ymax) / 2.0) / h
    ww = (xmax - xmin) / w
    hh = (ymax - ymin) / h
    return f"{clase_idx} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}"


def cargar_muestras(dir_img, dir_ann, prefijo):
    muestras = []
    for ruta_xml in glob.glob(os.path.join(dir_ann, "*.xml")):
        base = os.path.splitext(os.path.basename(ruta_xml))[0]
        ruta_img = os.path.join(dir_img, base + ".jpg")
        if not os.path.exists(ruta_img):
            continue
        muestras.append({"nombre_salida": f"{prefijo}_{base}", "ruta_imagen": ruta_img, "cajas": parsear_voc(ruta_xml)})
    return muestras


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
            lineas = [caja_a_yolo(0, c[1], c[2], c[3], c[4], w, h) for c in m["cajas"]]
            with open(ruta_lbl_dst, "w") as f:
                f.write("\n".join(lineas))
    ruta_yaml = os.path.join(dir_salida, "data.yaml")
    with open(ruta_yaml, "w") as f:
        f.write(f"path: {dir_salida}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: WF\n")
    return ruta_yaml


def sobremuestrear_propio(train):
    propio = [m for m in train if m.get("grupo") == "propio"]
    resto = [m for m in train if m.get("grupo") != "propio"]
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


def mascara_elipse_suave(w, h):
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    margen = max(1, int(min(w, h) * 0.08))
    d.ellipse((margen, margen, w - margen, h - margen), fill=255)
    return m.filter(ImageFilter.GaussianBlur(max(1, int(min(w, h) * 0.12))))


def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def detectar_area_trampa(imagen_pil):
    arr = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([32, 255, 255]))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    cnt = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(cnt) < (imagen_pil.width * imagen_pil.height * 0.05):
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    margen = int(min(w, h) * 0.04)
    return [x + margen, y + margen, x + w - margen, y + h - margen]


def area_desde_cajas(cajas_originales, w_img, h_img, expansion=0.15):
    if not cajas_originales:
        return None
    xs0 = [c[1] for c in cajas_originales]
    ys0 = [c[2] for c in cajas_originales]
    xs1 = [c[3] for c in cajas_originales]
    ys1 = [c[4] for c in cajas_originales]
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    w, h = x1 - x0, y1 - y0
    return [max(0, x0 - w * expansion), max(0, y0 - h * expansion), min(w_img, x1 + w * expansion), min(h_img, y1 + h * expansion)]


def encontrar_ubicacion(cajas_existentes, w_img, h_img, w_crop, h_crop, area_trampa):
    if area_trampa is not None:
        tx0, ty0, tx1, ty1 = area_trampa
        x_min, x_max = tx0, max(tx0, tx1 - w_crop)
        y_min, y_max = ty0, max(ty0, ty1 - h_crop)
        if x_max <= x_min or y_max <= y_min:
            return None
    else:
        x_min, x_max = 0, max(0, w_img - w_crop)
        y_min, y_max = 0, max(0, h_img - h_crop)
    for _ in range(MAX_INTENTOS_UBICACION):
        x0 = random.randint(int(x_min), int(x_max))
        y0 = random.randint(int(y_min), int(y_max))
        cand = [x0, y0, x0 + w_crop, y0 + h_crop]
        if all(iou(cand, c) <= SOLAPAMIENTO_MAX for c in cajas_existentes):
            return x0, y0
    return None


def ajustar_brillo(recorte, fondo, x0, y0):
    w, h = recorte.size
    parche = fondo.crop((x0, y0, x0 + w, y0 + h)).convert("L")
    bf = np.array(parche, dtype=np.float32).mean()
    br = np.array(recorte.convert("L"), dtype=np.float32).mean()
    if br < 1:
        return recorte
    factor = np.clip(bf / br, 0.6, 1.6)
    arr = np.clip(np.array(recorte, dtype=np.float32) * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def cargar_recortes_propio_train(train_propio):
    recortes = []
    for muestra in train_propio:
        cajas = muestra["cajas"]
        if not cajas:
            continue
        indices = list(range(len(cajas)))
        random.shuffle(indices)
        indices = indices[:MAX_RECORTES_POR_IMAGEN_PROPIO]
        try:
            with Image.open(muestra["ruta_imagen"]) as img:
                img = img.convert("RGB")
                for i in indices:
                    _, xmin, ymin, xmax, ymax = cajas[i]
                    w, h = xmax - xmin, ymax - ymin
                    if w <= 0 or h <= 0:
                        continue
                    px, py = w * PADDING_RECORTE, h * PADDING_RECORTE
                    x0, y0 = max(0, int(xmin - px)), max(0, int(ymin - py))
                    x1, y1 = min(img.width, int(xmax + px)), min(img.height, int(ymax + py))
                    if x1 - x0 < 6 or y1 - y0 < 6:
                        continue
                    recortes.append(img.crop((x0, y0, x1, y1)).copy())
        except Exception as e:
            print(f"Error leyendo {muestra['ruta_imagen']}: {e}")
    return recortes


def generar_variante(ruta_img_fondo, cajas_originales, recortes_propio):
    with Image.open(ruta_img_fondo) as fondo:
        fondo = fondo.convert("RGB").copy()
    w_img, h_img = fondo.size
    cajas_actuales = [list(c[1:]) for c in cajas_originales]
    nuevas = []
    area_trampa = detectar_area_trampa(fondo) or area_desde_cajas(cajas_originales, w_img, h_img)
    n_moscas = random.randint(MOSCAS_POR_VARIANTE_MIN, MOSCAS_POR_VARIANTE_MAX)
    for _ in range(n_moscas):
        recorte = random.choice(recortes_propio)
        w_crop, h_crop = recorte.size
        if w_crop >= w_img or h_crop >= h_img:
            continue
        ubi = encontrar_ubicacion(cajas_actuales, w_img, h_img, w_crop, h_crop, area_trampa)
        if ubi is None:
            continue
        x0, y0 = ubi
        recorte_aj = ajustar_brillo(recorte, fondo, x0, y0)
        fondo.paste(recorte_aj, (x0, y0), mascara_elipse_suave(w_crop, h_crop))
        caja_nueva = [x0 + w_crop * 0.15, y0 + h_crop * 0.15, x0 + w_crop * 0.85, y0 + h_crop * 0.85]
        cajas_actuales.append(caja_nueva)
        nuevas.append(caja_nueva)
    return fondo, [c[1:] for c in cajas_originales] + nuevas


def escribir_voc_xml(ruta_xml, nombre_archivo, w, h, cajas):
    lineas = ["<annotation>", f"  <filename>{nombre_archivo}</filename>", "  <size>",
              f"    <width>{w}</width>", f"    <height>{h}</height>", "    <depth>3</depth>", "  </size>"]
    for xmin, ymin, xmax, ymax in cajas:
        lineas += ["  <object>", "    <name>WF</name>", "    <bndbox>",
                   f"      <xmin>{int(xmin)}</xmin>", f"      <ymin>{int(ymin)}</ymin>",
                   f"      <xmax>{int(xmax)}</xmax>", f"      <ymax>{int(ymax)}</ymax>",
                   "    </bndbox>", "  </object>"]
    lineas.append("</annotation>")
    with open(ruta_xml, "w") as f:
        f.write("\n".join(lineas))


def generar_selfcp_fold(muestras_train_propio, dir_salida, semilla):
    random.seed(semilla)
    dir_img = os.path.join(dir_salida, "images")
    dir_ann = os.path.join(dir_salida, "annotations")
    os.makedirs(dir_img, exist_ok=True)
    os.makedirs(dir_ann, exist_ok=True)
    recortes_propio = cargar_recortes_propio_train(muestras_train_propio)
    print(f"  {len(recortes_propio)} recortes propios (tope {MAX_RECORTES_POR_IMAGEN_PROPIO}/imagen)")
    muestras_cp = []
    for m in muestras_train_propio:
        cajas_originales = [["WF"] + c[1:] for c in m["cajas"]]
        for n in range(VARIANTES_POR_IMAGEN):
            compuesta, cajas = generar_variante(m["ruta_imagen"], cajas_originales, recortes_propio)
            nombre = f"{m['nombre_salida']}_cp{n}"
            compuesta.save(os.path.join(dir_img, nombre + ".jpg"), "JPEG", quality=95)
            escribir_voc_xml(os.path.join(dir_ann, nombre + ".xml"), nombre + ".jpg", compuesta.width, compuesta.height, cajas)
            muestras_cp.append({
                "nombre_salida": f"cp_{nombre}", "ruta_imagen": os.path.join(dir_img, nombre + ".jpg"),
                "cajas": [["WF", *c] for c in cajas], "grupo": "propio",
            })
    return muestras_cp


def evaluar(modelo_path, data_yaml, nombre_run):
    modelo = YOLO(modelo_path)
    metricas = modelo.val(data=data_yaml, split="test", imgsz=IMGSZ, iou=IOU_EVAL, device=0, workers=0,
                           project=rf"{RAIZ}\yolo_runs", name=nombre_run, verbose=False)
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"P": p, "R": r, "mAP50": metricas.box.map50, "F1": f1}


if __name__ == "__main__":
    print("--- Cargando muestras publico (fijo, igual que siempre) y propio (60 img) ---")
    muestras_publico = cargar_muestras(DIR_MD121_IMG, DIR_MD121_ANN, "md121")
    for m in muestras_publico:
        m["grupo"] = "publico"
    muestras_propio = cargar_muestras(DIR_PROPIO_IMG, DIR_PROPIO_ANN, "propio")
    for m in muestras_propio:
        m["grupo"] = "propio"
    print(f"Publico: {len(muestras_publico)} img. Propio: {len(muestras_propio)} img.")

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
    print(f"5 folds de {tam_fold} imagenes propias (identico split al k-fold v2 anterior)")

    resultados_folds = []
    ruta_parciales = rf"{RAIZ}\kfold_resultados_parciales_v3.json"
    folds_ya_hechos = set()
    if os.path.exists(ruta_parciales):
        with open(ruta_parciales) as f:
            resultados_folds = json.load(f)
        folds_ya_hechos = {r["fold"] for r in resultados_folds}
        print(f"--- Reanudando: folds ya completados encontrados: {sorted(folds_ya_hechos)} ---")

    for fold_idx in range(N_FOLDS):
        nombre_fold = f"fold{fold_idx + 1}"
        if nombre_fold in folds_ya_hechos:
            print(f"\n--- {nombre_fold.upper()} ya completado, se omite ---")
            continue
        print(f"\n{'#' * 70}\n# {nombre_fold.upper()} ({fold_idx + 1}/{N_FOLDS}) -- self copy-paste, patience={PACIENCIA}\n{'#' * 70}")

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

        # reutiliza el dataset de test-propio-solo ya construido en la v2 (mismo split exacto)
        ruta_yaml_test_propio = rf"{RAIZ}\yolo_dataset_{nombre_fold}_test_propio_v2\data.yaml"

        print(f"\n--- Entrenando {nombre_fold} (self copy-paste, patience={PACIENCIA}) ---")
        modelo = YOLO("yolov8n.pt")
        modelo.train(
            data=ruta_yaml, epochs=EPOCAS, imgsz=IMGSZ, batch=BATCH, patience=PACIENCIA,
            device=0, workers=0, degrees=15.0, flipud=0.5,
            project=rf"{RAIZ}\yolo_runs", name=f"whitefly_yolov8n_{nombre_fold}_v3",
        )

        mejor_pt = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_{nombre_fold}_v3\weights\best.pt"
        print(f"\n--- Evaluando {nombre_fold}_v3 sobre su test propio held-out ({len(propio_test)} img) ---")
        resultado = evaluar(mejor_pt, ruta_yaml_test_propio, f"whitefly_{nombre_fold}_v3_eval_propio")
        print(f"{nombre_fold}_v3: P={resultado['P']:.4f} R={resultado['R']:.4f} F1={resultado['F1']:.4f}")
        resultados_folds.append({"fold": nombre_fold, **resultado})

        with open(rf"{RAIZ}\kfold_resultados_parciales_v3.json", "w") as f:
            json.dump(resultados_folds, f, indent=2)

    f1s = [r["F1"] for r in resultados_folds]
    print("\n" + "=" * 70)
    print("RESULTADO FINAL -- 5-FOLD CV v3 (SELF COPY-PASTE, patience=40)")
    print("=" * 70)
    for r in resultados_folds:
        print(f"  {r['fold']}: P={r['P']:.4f} R={r['R']:.4f} F1={r['F1']:.4f}")
    print("-" * 70)
    print(f"F1 propio media:            {np.mean(f1s):.4f}")
    print(f"F1 propio desv. estandar:   {np.std(f1s):.4f}")
    print(f"Para comparar: k-fold v2 (combinado-6 style, patience=20): media=0.6126 std=0.0777")
    print("=" * 70)

    with open(rf"{RAIZ}\kfold_resultados_finales_v3.json", "w") as f:
        json.dump({"folds": resultados_folds, "media": float(np.mean(f1s)), "std": float(np.std(f1s)),
                    "min": float(min(f1s)), "max": float(max(f1s))}, f, indent=2)
