"""Valida el ensemble WBF (v2 + v3) por fold, usando los 5 pares de modelos ya entrenados.
Cada fold evalua contra su propio test held-out (12 img), igual particion para ambas recetas.
Usa el umbral post-fusion optimo encontrado en el barrido (0.35 para propio)."""

import glob
import os

from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45
IOU_WBF = 0.55
UMBRAL_POST_FUSION = 0.35

FOLDS = {
    1: rf"{RAIZ}\yolo_runs\whitefly_yolov8n_fold1_v2_patience100\weights\best.pt",
    2: rf"{RAIZ}\yolo_runs\whitefly_yolov8n_fold2_v2\weights\best.pt",
    3: rf"{RAIZ}\yolo_runs\whitefly_yolov8n_fold3_v2\weights\best.pt",
    4: rf"{RAIZ}\yolo_runs\whitefly_yolov8n_fold4_v2\weights\best.pt",
    5: rf"{RAIZ}\yolo_runs\whitefly_yolov8n_fold5_v2\weights\best.pt",
}
FOLDS_V3 = {n: rf"{RAIZ}\yolo_runs\whitefly_yolov8n_fold{n}_v3\weights\best.pt" for n in range(1, 6)}

# solo los folds cuyo v3 SI existe en disco ahora mismo
FOLDS_DISPONIBLES = [n for n in range(1, 6) if os.path.exists(FOLDS_V3[n])]
print(f"Folds con v3 disponible ahora mismo: {FOLDS_DISPONIBLES}")


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


def emparejar(preds, gts, umbral_iou=0.45):
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
    resultados = {}
    for n in FOLDS_DISPONIBLES:
        modelo_v2 = YOLO(FOLDS[n])
        modelo_v3 = YOLO(FOLDS_V3[n])
        dir_img = rf"{RAIZ}\yolo_dataset_fold{n}_test_propio_v2\images\test"
        dir_lbl = rf"{RAIZ}\yolo_dataset_fold{n}_test_propio_v2\labels\test"
        rutas_img = sorted(glob.glob(os.path.join(dir_img, "*.jpg")))

        total_tp, total_fp, total_fn = 0, 0, 0
        for ruta_img in rutas_img:
            boxes_list, scores_list, labels_list = [], [], []
            img_w = img_h = None
            for modelo in (modelo_v2, modelo_v3):
                resultado = modelo.predict(source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS, device=0, verbose=False)[0]
                img_h, img_w = resultado.orig_shape
                cajas_norm, scores = [], []
                for box in resultado.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cajas_norm.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])
                    scores.append(float(box.conf[0]))
                boxes_list.append(cajas_norm)
                scores_list.append(scores)
                labels_list.append([0] * len(cajas_norm))

            cajas_f, scores_f, _ = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list, weights=None, iou_thr=IOU_WBF, skip_box_thr=0.0001,
            )
            preds = [
                ([x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h], s)
                for (x1, y1, x2, y2), s in zip(cajas_f, scores_f) if s >= UMBRAL_POST_FUSION
            ]

            nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
            gts = leer_gt_yolo(os.path.join(dir_lbl, nombre_base + ".txt"), img_w, img_h)
            tp, fp, fn = emparejar(preds, gts)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        resultados[n] = {"P": p, "R": r, "F1": f1}
        print(f"fold{n} ensemble: P={p:.4f} R={r:.4f} F1={f1:.4f}")

    import numpy as np
    f1s = [v["F1"] for v in resultados.values()]
    print(f"\nMedia F1 (folds {FOLDS_DISPONIBLES}): {np.mean(f1s):.4f}  std: {np.std(f1s):.4f}")
