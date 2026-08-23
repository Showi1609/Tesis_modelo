"""Diagnostico rapido: exporta los falsos positivos del fold1 (evaluado contra su propio test_propio
de 12 imagenes) para revisar a ojo si son errores reales del modelo o moscas reales sin etiquetar
(la misma sospecha que motivo la correccion de la Etapa 15, pero ahora sobre imagenes nunca auditadas).
Corre en CPU para no competir por VRAM con el fold3, que sigue entrenando en GPU."""

import csv
import glob
import os

from PIL import Image, ImageDraw
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_fold1\weights\best.pt"
DIR_IMG = rf"{RAIZ}\yolo_dataset_fold1_test_propio\images\test"
DIR_LBL = rf"{RAIZ}\yolo_dataset_fold1_test_propio\labels\test"
DIR_SALIDA = rf"{RAIZ}\diagnostico_fp_fold1"
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45
IOU_MATCH = 0.5
MARGEN_RECORTE = 40


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
    os.makedirs(DIR_SALIDA, exist_ok=True)
    modelo = YOLO(MODELO)
    rutas_img = sorted(glob.glob(os.path.join(DIR_IMG, "*.jpg")))

    filas_manifest = []
    total_fp = 0
    total_pred = 0
    total_gt = 0

    for ruta_img in rutas_img:
        resultado = modelo.predict(
            source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS, device="cpu", verbose=False,
        )[0]
        img_h, img_w = resultado.orig_shape
        preds = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in resultado.boxes]
        total_pred += len(preds)

        nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
        ruta_lbl = os.path.join(DIR_LBL, nombre_base + ".txt")
        gts = leer_gt_yolo(ruta_lbl, img_w, img_h)
        total_gt += len(gts)

        preds_ordenadas = sorted(preds, key=lambda p: -p[1])
        gt_usados = [False] * len(gts)
        fps = []
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
            else:
                fps.append((caja_pred, score))

        if not fps:
            continue

        with Image.open(ruta_img) as img:
            img = img.convert("RGB")
            for idx, (caja, score) in enumerate(fps):
                x1, y1, x2, y2 = caja
                cx0 = max(0, int(x1 - MARGEN_RECORTE))
                cy0 = max(0, int(y1 - MARGEN_RECORTE))
                cx1 = min(img_w, int(x2 + MARGEN_RECORTE))
                cy1 = min(img_h, int(y2 + MARGEN_RECORTE))
                recorte = img.crop((cx0, cy0, cx1, cy1)).copy()
                draw = ImageDraw.Draw(recorte)
                draw.rectangle(
                    [x1 - cx0, y1 - cy0, x2 - cx0, y2 - cy0], outline=(255, 0, 0), width=2
                )
                nombre_salida = f"{nombre_base}_fp{idx:02d}_conf{score:.2f}.jpg"
                recorte.save(os.path.join(DIR_SALIDA, nombre_salida), "JPEG", quality=95)
                filas_manifest.append({
                    "crop": nombre_salida,
                    "imagen_origen": nombre_base + ".jpg",
                    "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                    "confianza": round(score, 3),
                })
                total_fp += 1

    ruta_manifest = os.path.join(DIR_SALIDA, "manifest_fp.csv")
    with open(ruta_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop", "imagen_origen", "x1", "y1", "x2", "y2", "confianza"])
        writer.writeheader()
        writer.writerows(filas_manifest)

    print(f"Predicciones totales: {total_pred}  GT etiquetadas: {total_gt}")
    print(f"=== {total_fp} falsos positivos exportados a {DIR_SALIDA} ===")
    print(f"Manifest: {ruta_manifest}")
