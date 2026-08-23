"""
Auditoria de falsos positivos de alta confianza sobre el test propio (combinado-6, conf=0.25,
iou=0.45). Exporta un recorte por cada FP (prediccion sin match contra ground truth, IoU<0.5)
con confianza > UMBRAL_AUDITORIA, para revisarlos visualmente uno por uno.

Pregunta que responde: cuantos de esos "falsos positivos" son en realidad moscas reales que
quedaron sin etiquetar (lo que significaria que la precision real en propio es mas alta que el
58.8% reportado), y cuantos son errores genuinos del modelo.
"""

import glob
import os

from PIL import Image, ImageDraw
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt"
DIR_IMG = rf"{RAIZ}\yolo_dataset_desglose\images\test_propio"
DIR_LBL = rf"{RAIZ}\yolo_dataset_desglose\labels\test_propio"
DIR_SALIDA = rf"{RAIZ}\auditoria_fp_propio"
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45
IOU_MATCH = 0.5
UMBRAL_AUDITORIA = 0.6  # solo auditar FP con confianza alta (el modelo "seguro" de algo que no era)
MARGEN_RECORTE = 40  # px de contexto alrededor de cada caja


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

    total_fp_altos = 0
    for ruta_img in rutas_img:
        resultado = modelo.predict(
            source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS, device=0, verbose=False,
        )[0]
        img_h, img_w = resultado.orig_shape
        preds = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in resultado.boxes]

        nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
        ruta_lbl = os.path.join(DIR_LBL, nombre_base + ".txt")
        gts = leer_gt_yolo(ruta_lbl, img_w, img_h)

        # emparejar igual que en el bootstrap, pero guardando cuales quedan sin match
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

        fps_altos = [(c, s) for c, s in fps if s > UMBRAL_AUDITORIA]
        if not fps_altos:
            continue

        with Image.open(ruta_img) as img:
            img = img.convert("RGB")
            for idx, (caja, score) in enumerate(fps_altos):
                x1, y1, x2, y2 = caja
                cx0 = max(0, int(x1 - MARGEN_RECORTE))
                cy0 = max(0, int(y1 - MARGEN_RECORTE))
                cx1 = min(img_w, int(x2 + MARGEN_RECORTE))
                cy1 = min(img_h, int(y2 + MARGEN_RECORTE))
                recorte = img.crop((cx0, cy0, cx1, cy1)).copy()
                draw = ImageDraw.Draw(recorte)
                draw.rectangle(
                    [x1 - cx0, y1 - cy0, x2 - cx0, y2 - cy0], outline=(255, 0, 0), width=3
                )
                nombre_salida = f"{nombre_base}_fp{idx}_conf{score:.2f}.jpg"
                recorte.save(os.path.join(DIR_SALIDA, nombre_salida), "JPEG", quality=95)
                total_fp_altos += 1
                print(f"  {nombre_salida}  (caja original: {[round(v) for v in caja]})")

    print(f"\n=== {total_fp_altos} falsos positivos con conf>{UMBRAL_AUDITORIA} exportados a {DIR_SALIDA} ===")
