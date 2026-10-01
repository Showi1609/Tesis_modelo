"""Auditoria (2026-09-12): mide la razon area_pred/area_GT sobre los verdaderos positivos de
cada candidato, y guarda el IoU de cada match, separado por dominio. Define "verdadero
positivo" con iou_match=0.1 (el umbral permisivo/RD-03) para tener una muestra representativa
de "cajas que el algoritmo efectivamente encontro sobre una mosca real", sin condicionar la
muestra al criterio estricto (que para el Candidato A dejaria casi sin datos el analisis).

No modifica ningun script existente -- reutiliza las clases de
candidato_a_publico_fix_exif_iou01.py para AMBOS dominios del Candidato A (el flag EXIF es
inocuo sobre propio, que no tiene ese tag)."""

import csv
import glob
import os

import numpy as np
from ultralytics import YOLO

import candidato_a_publico_fix_exif_iou01 as cand_a

RAIZ = r"C:\Users\jchag\Documents\TESIS"
IOU_MATCH = 0.1
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45

DOMINIOS_A = {
    "publico": (rf"{RAIZ}\md121\images", rf"{RAIZ}\md121\annotations"),
    "propio": (rf"{RAIZ}\Propio\Tesis.voc\imagess", rf"{RAIZ}\Propio\Tesis.voc\annotations"),
}
DOMINIOS_B = {
    "publico": (rf"{RAIZ}\yolo_dataset_desglose\images\test_publico",
                rf"{RAIZ}\yolo_dataset_desglose\labels\test_publico"),
    "propio": (rf"{RAIZ}\yolo_dataset_desglose\images\test_propio",
               rf"{RAIZ}\yolo_dataset_desglose\labels\test_propio"),
}
MODELO_B_PATH = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\weights\best.pt"


def area(caja):
    return max(0.0, caja[2] - caja[0]) * max(0.0, caja[3] - caja[1])


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


def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    return inter / (area(a) + area(b) - inter)


def procesar_candidato_a(dominio, dir_img, dir_ann):
    preprocesador = cand_a.PreprocesamientoMIPE(erosion_borde=0, margen_trampa=0.03, debug=False)
    detector = cand_a.DetectorUnificado(debug=False)
    evaluador = cand_a.EvaluadorDataset(umbral_iou=IOU_MATCH)
    registros = []

    for ruta_img in sorted(glob.glob(os.path.join(dir_img, "*.jpg"))):
        nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
        ruta_xml = os.path.join(dir_ann, nombre_base + ".xml")
        if not os.path.exists(ruta_xml):
            continue
        cajas_xml = evaluador.parsear_xml(ruta_xml)
        img_prep, escala, matriz, msg = preprocesador.ejecutar_pipeline(ruta_img)
        if img_prep is None:
            continue
        cajas_gt = evaluador.transformar_cajas_gt(cajas_xml, escala, matriz)
        _, _, cajas_pred = detector.detectar_y_dibujar(img_prep.copy())

        gt_usados = [False] * len(cajas_gt)
        for pred in cajas_pred:
            mejor_iou, mejor_j = 0, -1
            for j, gt in enumerate(cajas_gt):
                if gt_usados[j]:
                    continue
                v = iou_xyxy(pred, gt)
                if v > mejor_iou:
                    mejor_iou, mejor_j = v, j
            if mejor_iou >= IOU_MATCH:
                gt_usados[mejor_j] = True
                registros.append({
                    "candidato": "A", "dominio": dominio, "imagen": nombre_base,
                    "iou": mejor_iou, "area_pred": area(pred), "area_gt": area(cajas_gt[mejor_j]),
                    "ratio_area": area(pred) / area(cajas_gt[mejor_j]) if area(cajas_gt[mejor_j]) > 0 else None,
                })
    return registros


def procesar_candidato_b(dominio, dir_img, dir_lbl):
    modelo = YOLO(MODELO_B_PATH)
    registros = []

    for ruta_img in sorted(glob.glob(os.path.join(dir_img, "*.jpg"))):
        resultado = modelo.predict(source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS,
                                    device=0, verbose=False)[0]
        img_h, img_w = resultado.orig_shape
        preds = [box.xyxy[0].tolist() for box in resultado.boxes]
        nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
        gts = leer_gt_yolo(os.path.join(dir_lbl, nombre_base + ".txt"), img_w, img_h)

        gt_usados = [False] * len(gts)
        for pred in preds:
            mejor_iou, mejor_j = 0, -1
            for j, gt in enumerate(gts):
                if gt_usados[j]:
                    continue
                v = iou_xyxy(pred, gt)
                if v > mejor_iou:
                    mejor_iou, mejor_j = v, j
            if mejor_iou >= IOU_MATCH:
                gt_usados[mejor_j] = True
                registros.append({
                    "candidato": "B", "dominio": dominio, "imagen": nombre_base,
                    "iou": mejor_iou, "area_pred": area(pred), "area_gt": area(gts[mejor_j]),
                    "ratio_area": area(pred) / area(gts[mejor_j]) if area(gts[mejor_j]) > 0 else None,
                })
    return registros


if __name__ == "__main__":
    todos = []
    for dominio, (dir_img, dir_ann) in DOMINIOS_A.items():
        print(f"--- Candidato A, dominio {dominio} ---")
        regs = procesar_candidato_a(dominio, dir_img, dir_ann)
        print(f"  {len(regs)} verdaderos positivos (iou_match={IOU_MATCH})")
        todos.extend(regs)

    for dominio, (dir_img, dir_lbl) in DOMINIOS_B.items():
        print(f"--- Candidato B, dominio {dominio} ---")
        regs = procesar_candidato_b(dominio, dir_img, dir_lbl)
        print(f"  {len(regs)} verdaderos positivos (iou_match={IOU_MATCH})")
        todos.extend(regs)

    ruta_csv = rf"{RAIZ}\analisis_areas_iou_tp.csv"
    with open(ruta_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidato", "dominio", "imagen", "iou", "area_pred", "area_gt", "ratio_area"])
        writer.writeheader()
        writer.writerows(todos)
    print(f"\nGuardado: {ruta_csv} ({len(todos)} filas)")

    print(f"\n{'='*70}\nRESUMEN -- razon de area (mediana y cuartiles), solo Candidato A\n{'='*70}")
    for dominio in DOMINIOS_A:
        ratios = [r["ratio_area"] for r in todos if r["candidato"] == "A" and r["dominio"] == dominio and r["ratio_area"] is not None]
        if not ratios:
            print(f"{dominio}: sin datos")
            continue
        q1, mediana, q3 = np.percentile(ratios, [25, 50, 75])
        print(f"{dominio} (n={len(ratios)}): Q1={q1:.4f}  mediana={mediana:.4f}  Q3={q3:.4f}  "
              f"media={np.mean(ratios):.4f}")

    print(f"\n{'='*70}\nRESUMEN -- distribucion de IoU de los TP, por candidato y dominio\n{'='*70}")
    for candidato in ("A", "B"):
        for dominio in ("publico", "propio"):
            ious = [r["iou"] for r in todos if r["candidato"] == candidato and r["dominio"] == dominio]
            if not ious:
                continue
            q1, mediana, q3 = np.percentile(ious, [25, 50, 75])
            print(f"Candidato {candidato}, {dominio} (n={len(ious)}): "
                  f"Q1={q1:.4f} mediana={mediana:.4f} Q3={q3:.4f} media={np.mean(ious):.4f}")
