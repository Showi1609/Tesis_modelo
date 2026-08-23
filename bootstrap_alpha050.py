"""
Antes de reabrir la Etapa 9 (weight averaging, alpha=0.50) como candidato bajo la nueva metrica
de decision (F1 propio), hay que aplicarle el mismo bootstrap que a combinado-6 -- si no, estamos
comparando un punto sin incertidumbre (0.641) contra otro (0.6325) como si la diferencia fuera
real, cuando ya sabemos que el ruido de 9 imagenes es de +-0.09 aprox.
"""

import glob
import os
import random

import numpy as np
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_soup\weights\soup_a0.50.pt"
DIR_IMG = rf"{RAIZ}\yolo_dataset_desglose\images\test_propio"
DIR_LBL = rf"{RAIZ}\yolo_dataset_desglose\labels\test_propio"
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45
IOU_MATCH = 0.5
N_BOOTSTRAP = 1000
SEMILLA = 42


def leer_gt_yolo(ruta_txt, img_w, img_h):
    cajas = []
    if not os.path.exists(ruta_txt):
        return cajas
    with open(ruta_txt) as f:
        for linea in f:
            partes = linea.split()
            if len(partes) < 5:
                continue
            _, cx, cy, w, h = map(float, partes[:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            cajas.append([x1, y1, x2, y2])
    return cajas


def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def emparejar(preds, gts, umbral_iou=IOU_MATCH):
    preds_ordenadas = sorted(preds, key=lambda p: -p[1])
    gt_usados = [False] * len(gts)
    tp = 0
    for caja_pred, _score in preds_ordenadas:
        mejor_iou, mejor_j = 0, -1
        for j, gt in enumerate(gts):
            if gt_usados[j]:
                continue
            v = iou(caja_pred, gt)
            if v > mejor_iou:
                mejor_iou, mejor_j = v, j
        if mejor_iou >= umbral_iou:
            gt_usados[mejor_j] = True
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


if __name__ == "__main__":
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)

    modelo = YOLO(MODELO)
    rutas_img = sorted(glob.glob(os.path.join(DIR_IMG, "*.jpg")))

    por_imagen = []
    for ruta_img in rutas_img:
        resultado = modelo.predict(
            source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS, device=0, verbose=False,
        )[0]
        img_h, img_w = resultado.orig_shape
        preds = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in resultado.boxes]
        nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
        gts = leer_gt_yolo(os.path.join(DIR_LBL, nombre_base + ".txt"), img_w, img_h)
        tp, fp, fn = emparejar(preds, gts)
        por_imagen.append((tp, fp, fn))

    tp_total = sum(t for t, _, _ in por_imagen)
    fp_total = sum(f for _, f, _ in por_imagen)
    fn_total = sum(n for _, _, n in por_imagen)
    p0 = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    r0 = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1_puntual = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0

    n = len(por_imagen)
    f1s = []
    for _ in range(N_BOOTSTRAP):
        muestra = [por_imagen[random.randrange(n)] for _ in range(n)]
        tp_b = sum(t for t, _, _ in muestra)
        fp_b = sum(f for _, f, _ in muestra)
        fn_b = sum(nn for _, _, nn in muestra)
        p_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
        r_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
        f1s.append(2 * p_b * r_b / (p_b + r_b) if (p_b + r_b) > 0 else 0)

    f1s = np.array(f1s)
    print("=" * 60)
    print("BOOTSTRAP -- F1 propio (alpha=0.50, Etapa 9)")
    print("=" * 60)
    print(f"Punto central:                  F1 = {f1_puntual:.4f}")
    print(f"Bootstrap media:                F1 = {f1s.mean():.4f}")
    print(f"Bootstrap desviacion estandar:  {f1s.std():.4f}")
    print(f"Intervalo de confianza 95%:     [{np.percentile(f1s, 2.5):.4f}, {np.percentile(f1s, 97.5):.4f}]")
    print("=" * 60)
    print(f"\nPara comparar: combinado-6 tenia F1 puntual=0.6217-0.6325, CI95=[0.4881, 0.6682]")
