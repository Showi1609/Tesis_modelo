"""Reevalua combinado6_v2 (propio y publico) con un umbral de EMPAREJAMIENTO GT manual (no el
NMS de Ultralytics), para poder comparar directamente contra Candidato A con el mismo criterio
de matching en ambos sentidos (permisivo iou_match=0.1 y estricto iou_match=0.45)."""

import glob
import os

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\weights\best.pt"
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45  # NMS, sin cambios -- esto NO es el criterio de matching contra GT


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


def evaluar_con_match(modelo, dir_img, dir_lbl, iou_match):
    rutas_img = sorted(glob.glob(os.path.join(dir_img, "*.jpg")))
    total_tp, total_fp, total_fn = 0, 0, 0
    for ruta_img in rutas_img:
        resultado = modelo.predict(
            source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS, device=0, verbose=False,
        )[0]
        img_h, img_w = resultado.orig_shape
        preds = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in resultado.boxes]
        nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
        gts = leer_gt_yolo(os.path.join(dir_lbl, nombre_base + ".txt"), img_w, img_h)

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
            if mejor_iou >= iou_match:
                gt_usados[mejor_j] = True
                tp += 1
        fp = len(preds) - tp
        fn = len(gts) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


if __name__ == "__main__":
    modelo = YOLO(MODELO)

    subsets = [
        ("propio", rf"{RAIZ}\yolo_dataset_desglose\images\test_propio", rf"{RAIZ}\yolo_dataset_desglose\labels\test_propio"),
        ("publico", rf"{RAIZ}\yolo_dataset_desglose\images\test_publico", rf"{RAIZ}\yolo_dataset_desglose\labels\test_publico"),
    ]

    print("=" * 60)
    print("COMBINADO-6_V2 -- emparejamiento manual, mismo criterio que Candidato A")
    print("=" * 60)
    for nombre, dir_img, dir_lbl in subsets:
        for iou_match in (0.1, 0.45):
            p, r, f1 = evaluar_con_match(modelo, dir_img, dir_lbl, iou_match)
            print(f"[{nombre}] iou_match={iou_match}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
