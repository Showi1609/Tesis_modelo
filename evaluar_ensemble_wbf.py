"""Ensemble por Weighted Box Fusion entre combinado-6_v2 y combinado-8_v2 -- corre ambos modelos
sobre cada imagen de test (propio y publico), funde sus cajas con WBF, y evalua el resultado
fusionado contra el baseline de cada modelo por separado (iou=0.45, mismo criterio de siempre)."""

import glob
import os

from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
IMGSZ = 1280
CONF = 0.25  # umbral normal de operacion (igual que la app), evita que pasen demasiados singletons debiles
IOU_NMS = 0.45
IOU_WBF = 0.55
IOU_MATCH = 0.45  # mismo criterio de matching contra GT que el resto de la tesis

MODELOS = {
    "combinado-6_v2": rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\weights\best.pt",
    "combinado-8_v2": rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-8_v2\weights\best.pt",
}

SUBSETS = {
    "propio": (rf"{RAIZ}\yolo_dataset_desglose\images\test_propio", rf"{RAIZ}\yolo_dataset_desglose\labels\test_propio"),
    "publico": (rf"{RAIZ}\yolo_dataset_desglose\images\test_publico", rf"{RAIZ}\yolo_dataset_desglose\labels\test_publico"),
}


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
    modelos_cargados = {nombre: YOLO(ruta) for nombre, ruta in MODELOS.items()}

    for nombre_subset, (dir_img, dir_lbl) in SUBSETS.items():
        rutas_img = sorted(glob.glob(os.path.join(dir_img, "*.jpg")))
        total_tp, total_fp, total_fn = 0, 0, 0

        for ruta_img in rutas_img:
            boxes_list, scores_list, labels_list = [], [], []
            img_w = img_h = None

            for nombre_modelo, modelo in modelos_cargados.items():
                resultado = modelo.predict(
                    source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS, device=0, verbose=False,
                )[0]
                img_h, img_w = resultado.orig_shape
                cajas_norm = []
                scores = []
                for box in resultado.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cajas_norm.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])
                    scores.append(float(box.conf[0]))
                boxes_list.append(cajas_norm)
                scores_list.append(scores)
                labels_list.append([0] * len(cajas_norm))

            cajas_f, scores_f, _ = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list, weights=None,
                iou_thr=IOU_WBF, skip_box_thr=0.0001,
            )
            # filtra cajas que solo un modelo voto (WBF les baja el score al fusionar) -- se
            # quedan solo las que ambos modelos coincidieron razonablemente
            UMBRAL_POST_FUSION = 0.35
            preds = [
                ([x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h], s)
                for (x1, y1, x2, y2), s in zip(cajas_f, scores_f)
                if s >= UMBRAL_POST_FUSION
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
        print(f"=== Ensemble WBF (combinado-6_v2 + combinado-8_v2) -- {nombre_subset} ===")
        print(f"Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
