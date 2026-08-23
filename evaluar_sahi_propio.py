"""
Etapa 16: SAHI (Slicing Aided Hyper Inference) sobre combinado-6/best.pt, sin reentrenar nada.
En vez de aplastar la imagen completa (hasta 4000x2252px) a 1280px para inferencia -- perdiendo
resolucion en un objeto ya de por si diminuto -- se corta la imagen en tiles de 1280x1280 con
solape, se corre inferencia en cada tile a su resolucion nativa, y se funden los resultados.

Se evalua sobre las 9 imagenes ORIGINALES (no la copia redimensionada de yolo_dataset_desglose)
para aprovechar toda la resolucion nativa, contra las etiquetas ya corregidas (Etapa 15).
Comparacion directa contra el baseline sin SAHI: P=80.6% R=71.5% F1=0.7578.
"""

import glob
import os
import sys

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

sys.path.insert(0, r"C:\Users\jchag\Documents\TESIS")
import yolo_entrenamiento as yt

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt"
DIR_ANOTACIONES = rf"{RAIZ}\Propio\Tesis.voc\annotations"
DIR_IMG_ORIGINAL = rf"{RAIZ}\Propio\Tesis.voc\imagess"

CONF = 0.25       # igual que la app
IOU_NMS = 0.45    # igual que la app, se usa para fundir los tiles
IOU_MATCH = 0.5
SLICE = 1280
OVERLAP = 0.2


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
    print("--- Determinando el split propio de test (mismo de siempre) ---")
    muestras_por_fuente = yt.recolectar_muestras()
    train, val, test = yt.dividir_train_val_test(
        muestras_por_fuente, yt.PROPORCION_VAL, yt.PROPORCION_TEST, yt.SEMILLA
    )
    test_propio = [m for m in test if m["grupo"] == "propio"]
    print(f"{len(test_propio)} imagenes de test propio\n")

    print("--- Cargando modelo con SAHI ---")
    modelo = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODELO,
        confidence_threshold=CONF,
        device="cuda:0",
    )

    por_imagen = []
    for muestra in test_propio:
        nombre_base = os.path.splitext(os.path.basename(muestra["ruta_imagen"]))[0]
        resultado = get_sliced_prediction(
            muestra["ruta_imagen"],
            modelo,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            postprocess_type="NMS",
            postprocess_match_threshold=IOU_NMS,
            verbose=0,
        )
        preds = [
            (
                [op.bbox.minx, op.bbox.miny, op.bbox.maxx, op.bbox.maxy],
                op.score.value,
            )
            for op in resultado.object_prediction_list
        ]
        gts = [c[1:] for c in muestra["cajas"]]  # ya en coords de la imagen original
        tp, fp, fn = emparejar(preds, gts)
        por_imagen.append((tp, fp, fn))
        p_img = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_img = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  {nombre_base}: {len(gts)} reales, {len(preds)} predichas (SAHI) -> "
              f"TP={tp} FP={fp} FN={fn} (P={p_img:.2f} R={r_img:.2f})")

    tp_t = sum(t for t, _, _ in por_imagen)
    fp_t = sum(f for _, f, _ in por_imagen)
    fn_t = sum(n for _, _, n in por_imagen)
    p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    print("\n" + "=" * 60)
    print("SAHI (tiles 1280x1280, solape 20%) vs. BASELINE (imagen completa a 1280px)")
    print("=" * 60)
    print(f"SAHI:      P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    print(f"Baseline:  P=0.8065  R=0.7147  F1=0.7578")
    print("=" * 60)
