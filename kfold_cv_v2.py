"""
Etapa 19b: repite el 5-fold CV (Etapa 18) pero con las etiquetas propias corregidas
(2026-08-21, correccion via Roboflow de ~11 imagenes masivamente sub-etiquetadas que el
primer k-fold destapo). Misma estructura de folds (misma semilla=42, mismos 5 grupos de 12
imagenes) para poder comparar fold-por-fold contra el resultado viejo:
  fold1=0.5606 fold2=0.5494 fold3=0.3927 fold4=0.4598 fold5=0.5175, media=0.4960+-0.0624

Todas las carpetas de salida llevan sufijo _v2 para no pisar el k-fold anterior (que queda
como referencia historica del "antes" de la correccion de etiquetas).
"""

import glob
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DIR_MD121_IMG = rf"{RAIZ}\md121\images"
DIR_MD121_ANN = rf"{RAIZ}\md121\annotations"
DIR_PROPIO_IMG = rf"{RAIZ}\Propio\Tesis.voc\imagess"
DIR_PROPIO_ANN = rf"{RAIZ}\Propio\Tesis.voc\annotations"

CLASES = ["WF"]
N_FOLDS = 5
SEMILLA = 42
MAX_DIM_DATASET = 1920
IMGSZ = 1280
EPOCAS = 100
BATCH = 4
PACIENCIA = 20
IOU_EVAL = 0.45

VARIANTES_POR_IMAGEN = 3
MOSCAS_POR_VARIANTE_MIN = 3
MOSCAS_POR_VARIANTE_MAX = 8
PADDING_RECORTE = 0.15
MAX_INTENTOS_UBICACION = 30
SOLAPAMIENTO_MAX = 0.05


def parsear_voc(ruta_xml):
    cajas = []
    try:
        root = ET.parse(ruta_xml).getroot()
        for obj in root.findall("object"):
            nombre_clase = obj.find("name").text
            if nombre_clase not in CLASES:
                continue
            b = obj.find("bndbox")
            cajas.append([
                nombre_clase,
                float(b.find("xmin").text), float(b.find("ymin").text),
                float(b.find("xmax").text), float(b.find("ymax").text),
            ])
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
        muestras.append({
            "nombre_salida": f"{prefijo}_{base}",
            "ruta_imagen": ruta_img,
            "cajas": parsear_voc(ruta_xml),
        })
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
        f.write(f"path: {dir_salida}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n  0: WF\n")
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


def cargar_recortes_publico():
    recortes = []
    for ruta_xml in glob.glob(os.path.join(DIR_MD121_ANN, "*.xml")):
        base = os.path.splitext(os.path.basename(ruta_xml))[0]
        ruta_img = os.path.join(DIR_MD121_IMG, base + ".jpg")
        if not os.path.exists(ruta_img):
            continue
        cajas = parsear_voc(ruta_xml)
        if not cajas:
            continue
        try:
            with Image.open(ruta_img) as img:
                img = img.convert("RGB")
                for _, xmin, ymin, xmax, ymax in cajas:
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
            print(f"Error leyendo {ruta_img}: {e}")
    return recortes


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
    return [max(0, x0 - w * expansion), max(0, y0 - h * expansion),
            min(w_img, x1 + w * expansion), min(h_img, y1 + h * expansion)]


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


def generar_variante(ruta_img_fondo, cajas_originales, recortes_publico):
    with Image.open(ruta_img_fondo) as fondo:
        fondo = fondo.convert("RGB").copy()
    w_img, h_img = fondo.size
    cajas_actuales = [list(c[1:]) for c in cajas_originales]
    nuevas = []
    area_trampa = detectar_area_trampa(fondo) or area_desde_cajas(cajas_originales, w_img, h_img)
    n_moscas = random.randint(MOSCAS_POR_VARIANTE_MIN, MOSCAS_POR_VARIANTE_MAX)
    for _ in range(n_moscas):
        recorte = random.choice(recortes_publico)
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


def generar_copy_paste_fold(muestras_train_propio, recortes_publico, dir_salida, semilla):
    random.seed(semilla)
    dir_img = os.path.join(dir_salida, "images")
    dir_ann = os.path.join(dir_salida, "annotations")
    os.makedirs(dir_img, exist_ok=True)
    os.makedirs(dir_ann, exist_ok=True)
    muestras_cp = []
    for m in muestras_train_propio:
        cajas_originales = [["WF"] + c[1:] for c in m["cajas"]]
        for n in range(VARIANTES_POR_IMAGEN):
            compuesta, cajas = generar_variante(m["ruta_imagen"], cajas_originales, recortes_publico)
            nombre = f"{m['nombre_salida']}_cp{n}"
            compuesta.save(os.path.join(dir_img, nombre + ".jpg"), "JPEG", quality=95)
            escribir_voc_xml(os.path.join(dir_ann, nombre + ".xml"), nombre + ".jpg",
                              compuesta.width, compuesta.height, cajas)
            muestras_cp.append({
                "nombre_salida": f"cp_{nombre}",
                "ruta_imagen": os.path.join(dir_img, nombre + ".jpg"),
                "cajas": [["WF", *c] for c in cajas],
                "grupo": "propio",
            })
    return muestras_cp


def evaluar(modelo_path, data_yaml, nombre_run):
    modelo = YOLO(modelo_path)
    metricas = modelo.val(
        data=data_yaml, split="test", imgsz=IMGSZ, iou=IOU_EVAL, device=0, workers=0,
        project=rf"{RAIZ}\yolo_runs", name=nombre_run, verbose=False,
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"P": p, "R": r, "mAP50": metricas.box.map50, "F1": f1}


if __name__ == "__main__":
    print("--- Cargando muestras publico (fijo en todos los folds) y propio (60 img, CORREGIDAS) ---")
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
    print(f"Publico fijo -> train={len(pub_train)} val={len(pub_val)} test={len(pub_test)}")

    rng_folds = random.Random(SEMILLA)
    copia_propio = muestras_propio[:]
    rng_folds.shuffle(copia_propio)
    tam_fold = len(copia_propio) // N_FOLDS
    folds = [copia_propio[i * tam_fold:(i + 1) * tam_fold] for i in range(N_FOLDS)]
    print(f"5 folds de {tam_fold} imagenes propias cada uno (misma particion que el k-fold anterior)")

    print("\n--- Cargando pool de recortes publico (una sola vez, se reusa en todos los folds) ---")
    recortes_publico = cargar_recortes_publico()
    print(f"{len(recortes_publico)} recortes publicos disponibles")

    resultados_folds = []

    for fold_idx in range(N_FOLDS):
        nombre_fold = f"fold{fold_idx + 1}"
        print(f"\n{'#' * 70}\n# {nombre_fold.upper()} ({fold_idx + 1}/{N_FOLDS}) -- v2 (etiquetas corregidas)\n{'#' * 70}")

        propio_test = folds[fold_idx]
        resto = [img for j, f in enumerate(folds) if j != fold_idx for img in f]
        rng_val = random.Random(SEMILLA + fold_idx)
        rng_val.shuffle(resto)
        n_val = max(1, round(len(resto) * 0.15))
        propio_val = resto[:n_val]
        propio_train = resto[n_val:]
        print(f"Propio -> train={len(propio_train)} val={len(propio_val)} test={len(propio_test)}")

        dir_cp = rf"{RAIZ}\Propio_copypaste_{nombre_fold}_v2"
        print(f"--- Generando copy-paste para {nombre_fold} (v2) ---")
        muestras_cp = generar_copy_paste_fold(propio_train, recortes_publico, dir_cp, SEMILLA + fold_idx)
        print(f"{len(muestras_cp)} imagenes sinteticas generadas")

        train = pub_train + propio_train + muestras_cp
        val = pub_val + propio_val
        test = pub_test + propio_test

        train, factor = sobremuestrear_propio(train)
        print(f"Sobremuestreo factor x{factor}, train total = {len(train)} imagenes")

        dir_dataset = rf"{RAIZ}\yolo_dataset_{nombre_fold}_v2"
        ruta_yaml = construir_dataset_yolo(train, val, test, dir_dataset)
        print(f"Dataset del {nombre_fold} (v2) listo en {dir_dataset}")

        dir_dataset_test_propio = rf"{RAIZ}\yolo_dataset_{nombre_fold}_test_propio_v2"
        construir_dataset_yolo(propio_test, propio_test, propio_test, dir_dataset_test_propio)
        ruta_yaml_test_propio = os.path.join(dir_dataset_test_propio, "data.yaml")

        print(f"\n--- Entrenando {nombre_fold} (v2) ---")
        modelo = YOLO("yolov8n.pt")
        modelo.train(
            data=ruta_yaml, epochs=EPOCAS, imgsz=IMGSZ, batch=BATCH, patience=PACIENCIA,
            device=0, workers=0, degrees=15.0, flipud=0.5,
            project=rf"{RAIZ}\yolo_runs", name=f"whitefly_yolov8n_{nombre_fold}_v2",
        )

        mejor_pt = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_{nombre_fold}_v2\weights\best.pt"
        print(f"\n--- Evaluando {nombre_fold} (v2) sobre su test propio held-out ({len(propio_test)} img) ---")
        resultado = evaluar(mejor_pt, ruta_yaml_test_propio, f"whitefly_{nombre_fold}_v2_eval_propio")
        print(f"{nombre_fold} (v2): P={resultado['P']:.4f} R={resultado['R']:.4f} F1={resultado['F1']:.4f}")
        resultados_folds.append({"fold": nombre_fold, **resultado})

        with open(rf"{RAIZ}\kfold_resultados_parciales_v2.json", "w") as f:
            json.dump(resultados_folds, f, indent=2)

    f1s = [r["F1"] for r in resultados_folds]
    print("\n" + "=" * 70)
    print("RESULTADO FINAL -- 5-FOLD CV v2 (ETIQUETAS CORREGIDAS) SOBRE LAS 60 IMAGENES PROPIAS")
    print("=" * 70)
    for r in resultados_folds:
        print(f"  {r['fold']}: P={r['P']:.4f} R={r['R']:.4f} F1={r['F1']:.4f}")
    print("-" * 70)
    print(f"F1 propio media:            {np.mean(f1s):.4f}")
    print(f"F1 propio desv. estandar:   {np.std(f1s):.4f}")
    print(f"F1 propio min/max:          [{min(f1s):.4f}, {max(f1s):.4f}]")
    print(f"\nPara comparar: k-fold v1 (etiquetas viejas) dio media=0.4960 std=0.0624")
    print(f"Para comparar: combinado-6_v2 (split fijo de 9 img, etiquetas corregidas) dio F1 propio = 0.6705")
    print("=" * 70)

    with open(rf"{RAIZ}\kfold_resultados_finales_v2.json", "w") as f:
        json.dump({
            "folds": resultados_folds,
            "media": float(np.mean(f1s)),
            "std": float(np.std(f1s)),
            "min": float(min(f1s)),
            "max": float(max(f1s)),
        }, f, indent=2)
