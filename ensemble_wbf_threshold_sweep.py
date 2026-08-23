"""Corre WBF una sola vez por imagen (guardando todos los candidatos fusionados) y despues barre
el umbral post-fusion sin volver a correr los modelos -- igual patron que sahi_threshold_sweep.py."""

import glob
import os

from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45
IOU_WBF = 0.55
UMBRALES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

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
    modelos_cargados = {nombre: YOLO(ruta) for nombre, ruta in MODELOS.items()}

    for nombre_subset, (dir_img, dir_lbl) in SUBSETS.items():
        rutas_img = sorted(glob.glob(os.path.join(dir_img, "*.jpg")))
        todo_por_imagen = []  # (gts, [(caja, score), ...] ya fusionadas, sin filtrar)

        for ruta_img in rutas_img:
            boxes_list, scores_list, labels_list = [], [], []
            img_w = img_h = None
            for modelo in modelos_cargados.values():
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
            preds = [([x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h], s) for (x1, y1, x2, y2), s in zip(cajas_f, scores_f)]

            nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
            gts = leer_gt_yolo(os.path.join(dir_lbl, nombre_base + ".txt"), img_w, img_h)
            todo_por_imagen.append((gts, preds))

        print(f"\n=== Barrido umbral post-fusion -- {nombre_subset} ===")
        print(f"{'umbral':>7} | {'P':>8} | {'R':>8} | {'F1':>8}")
        mejor = (0, 0)
        for umbral in UMBRALES:
            tp_t, fp_t, fn_t = 0, 0, 0
            for gts, preds in todo_por_imagen:
                filtradas = [(c, s) for c, s in preds if s >= umbral]
                tp, fp, fn = emparejar(filtradas, gts)
                tp_t += tp
                fp_t += fp
                fn_t += fn
            p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > mejor[1]:
                mejor = (umbral, f1)
            print(f"{umbral:>7.2f} | {p:>8.4f} | {r:>8.4f} | {f1:>8.4f}")
        print(f"Mejor umbral: {mejor[0]:.2f} -> F1={mejor[1]:.4f}")
