"""
Bootstrap sobre el test propio (9 imagenes) para combinado-6/best.pt, con el conf/iou reales de
la app (conf=0.25, iou=0.45). El F1=0.6325 reportado hasta ahora es un punto unico sobre solo 9
imagenes -- aqui se cuantifica cuanta incertidumbre trae eso.

Metodologia:
  1. Correr inferencia una vez sobre las 9 imagenes de test_propio con combinado-6/best.pt.
  2. Emparejar predicciones vs. ground truth por imagen (greedy, IoU>=0.5) -> TP/FP/FN por imagen.
  3. Remuestrear las 9 imagenes CON reemplazo, 1000 veces; sumar TP/FP/FN del remuestreo;
     recalcular P/R/F1; guardar la distribucion.
  4. Reportar media, desviacion estandar e intervalo de confianza 95% (percentiles 2.5/97.5).

No reentrena nada -- son las mismas 9 imagenes de test held-out de siempre.
"""

import glob
import os
import random

import numpy as np
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt"
DIR_IMG = rf"{RAIZ}\yolo_dataset_desglose\images\test_propio"
DIR_LBL = rf"{RAIZ}\yolo_dataset_desglose\labels\test_propio"
IMGSZ = 1280
CONF = 0.25   # igual a la app
IOU_NMS = 0.45  # igual a la app
IOU_MATCH = 0.5  # umbral estandar para contar TP (igual que mAP50)
N_BOOTSTRAP = 1000
SEMILLA = 42


def leer_gt_yolo(ruta_txt, img_w, img_h):
    """Lee un label YOLO (clase cx cy w h normalizados) y devuelve cajas [x1,y1,x2,y2] en pixeles."""
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
    """Greedy matching por confianza descendente. Devuelve (tp, fp, fn) de la imagen."""
    preds_ordenadas = sorted(preds, key=lambda p: -p[1])  # por score desc
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
    print(f"{len(rutas_img)} imagenes de test propio")

    por_imagen = []  # lista de (tp, fp, fn) por imagen
    for ruta_img in rutas_img:
        resultado = modelo.predict(
            source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS, device=0, verbose=False,
        )[0]
        img_h, img_w = resultado.orig_shape
        preds = [
            (box.xyxy[0].tolist(), float(box.conf[0]))
            for box in resultado.boxes
        ]
        nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
        ruta_lbl = os.path.join(DIR_LBL, nombre_base + ".txt")
        gts = leer_gt_yolo(ruta_lbl, img_w, img_h)

        tp, fp, fn = emparejar(preds, gts)
        por_imagen.append((tp, fp, fn))
        p_img = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_img = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  {nombre_base}: {len(gts)} reales, {len(preds)} predichas -> TP={tp} FP={fp} FN={fn} "
              f"(P={p_img:.2f} R={r_img:.2f})")

    tp_total = sum(t for t, _, _ in por_imagen)
    fp_total = sum(f for _, f, _ in por_imagen)
    fn_total = sum(n for _, _, n in por_imagen)
    p_puntual = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    r_puntual = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1_puntual = 2 * p_puntual * r_puntual / (p_puntual + r_puntual) if (p_puntual + r_puntual) > 0 else 0
    print(f"\nPunto central (las 9 imagenes, sin remuestreo): P={p_puntual:.4f} R={r_puntual:.4f} F1={f1_puntual:.4f}")

    print(f"\n--- Bootstrap: remuestreando {len(por_imagen)} imagenes con reemplazo, {N_BOOTSTRAP} veces ---")
    f1s, ps, rs = [], [], []
    n = len(por_imagen)
    for _ in range(N_BOOTSTRAP):
        muestra = [por_imagen[random.randrange(n)] for _ in range(n)]
        tp_b = sum(t for t, _, _ in muestra)
        fp_b = sum(f for _, f, _ in muestra)
        fn_b = sum(nn for _, _, nn in muestra)
        p_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
        r_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
        f1_b = 2 * p_b * r_b / (p_b + r_b) if (p_b + r_b) > 0 else 0
        f1s.append(f1_b)
        ps.append(p_b)
        rs.append(r_b)

    f1s = np.array(f1s)
    print("\n" + "=" * 60)
    print("BOOTSTRAP -- F1 propio (combinado-6, conf=0.25, iou=0.45)")
    print("=" * 60)
    print(f"Punto central (9 img reales):   F1 = {f1_puntual:.4f}")
    print(f"Bootstrap media:                F1 = {f1s.mean():.4f}")
    print(f"Bootstrap desviacion estandar:  {f1s.std():.4f}")
    print(f"Intervalo de confianza 95%:     [{np.percentile(f1s, 2.5):.4f}, {np.percentile(f1s, 97.5):.4f}]")
    print(f"Rango min/max observado:        [{f1s.min():.4f}, {f1s.max():.4f}]")
    print("=" * 60)
