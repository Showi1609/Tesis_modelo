"""Corre el mismo diagnostico de falsos positivos (Etapa 18) sobre los 5 folds completos --
cada fold evalua su propio best.pt contra sus 12 imagenes de test held-out, cubriendo las 60
imagenes propias exactamente una vez entre los 5. Objetivo: encontrar imagenes atipicas
(pocas cajas GT pero muchas detecciones sin match) para revisar a mano en Roboflow."""

import csv
import glob
import os

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45
IOU_MATCH = 0.5


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


if __name__ == "__main__":
    filas = []

    for fold_idx in range(1, 6):
        nombre_fold = f"fold{fold_idx}"
        modelo_path = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_{nombre_fold}\weights\best.pt"
        dir_img = rf"{RAIZ}\yolo_dataset_{nombre_fold}_test_propio\images\test"
        dir_lbl = rf"{RAIZ}\yolo_dataset_{nombre_fold}_test_propio\labels\test"

        print(f"\n--- {nombre_fold} ---")
        modelo = YOLO(modelo_path)
        rutas_img = sorted(glob.glob(os.path.join(dir_img, "*.jpg")))

        for ruta_img in rutas_img:
            resultado = modelo.predict(
                source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS, device=0, verbose=False,
            )[0]
            img_h, img_w = resultado.orig_shape
            preds = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in resultado.boxes]

            nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
            ruta_lbl = os.path.join(dir_lbl, nombre_base + ".txt")
            gts = leer_gt_yolo(ruta_lbl, img_w, img_h)

            preds_ordenadas = sorted(preds, key=lambda p: -p[1])
            gt_usados = [False] * len(gts)
            tp, fp = 0, 0
            for caja_pred, score in preds_ordenadas:
                mejor_iou, mejor_j = 0, -1
                for j, gt in enumerate(gts):
                    if gt_usados[j]:
                        continue
                    v = iou(caja_pred, gt)
                    if v > mejor_iou:
                        mejor_iou, mejor_j = v, j
                if mejor_iou >= IOU_MATCH:
                    gt_usados[mejor_j] = True
                    tp += 1
                else:
                    fp += 1
            fn = len(gts) - tp

            # nombre original (2078, 2081, etc.) a partir de "propio_2078_jpg.rf...."
            nombre_original = nombre_base.replace("propio_", "").split("_jpg")[0]

            filas.append({
                "fold": nombre_fold,
                "imagen": nombre_original,
                "gt": len(gts),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ratio_fp_gt": round(fp / max(1, len(gts)), 2),
            })
            print(f"  {nombre_original}: GT={len(gts)} TP={tp} FP={fp} FN={fn}")

    filas.sort(key=lambda r: -r["fp"])
    ruta_csv = rf"{RAIZ}\diagnostico_fp_todos_folds.csv"
    with open(ruta_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "imagen", "gt", "tp", "fp", "fn", "ratio_fp_gt"])
        writer.writeheader()
        writer.writerows(filas)

    print(f"\n=== Guardado: {ruta_csv} ===")
    print("\n--- TOP 15 por cantidad de FP ---")
    for r in filas[:15]:
        print(f"  {r['imagen']} ({r['fold']}): GT={r['gt']} FP={r['fp']} FN={r['fn']} ratio_fp/gt={r['ratio_fp_gt']}")

    print("\n--- Casos con GT<=3 pero FP>=10 (sospecha fuerte de sub-etiquetado) ---")
    for r in filas:
        if r["gt"] <= 3 and r["fp"] >= 10:
            print(f"  {r['imagen']} ({r['fold']}): GT={r['gt']} FP={r['fp']} FN={r['fn']}")
