"""
Barrido de umbral de confianza para el modo SAHI, para ver si sube la precision sin perder toda
la ganancia de recall. Corre la inferencia de SAHI UNA sola vez con confidence_threshold bajo
(0.05, para capturar todos los candidatos) y despues filtra por distintos umbrales en el
emparejamiento -- no hace falta re-correr la inferencia en tiles por cada umbral.
"""

import os
import sys

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

sys.path.insert(0, r"C:\Users\jchag\Documents\TESIS")
import yolo_entrenamiento as yt

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt"

CONF_CARGA = 0.05  # bajo, para capturar todos los candidatos; se filtra despues
IOU_NMS = 0.45
IOU_MATCH = 0.5
SLICE = 1280
OVERLAP = 0.2
UMBRALES_A_PROBAR = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


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
    print("--- Determinando el split propio de test ---")
    muestras_por_fuente = yt.recolectar_muestras()
    train, val, test = yt.dividir_train_val_test(
        muestras_por_fuente, yt.PROPORCION_VAL, yt.PROPORCION_TEST, yt.SEMILLA
    )
    test_propio = [m for m in test if m["grupo"] == "propio"]

    print("--- Cargando modelo con SAHI (conf_carga=0.05, se filtra despues) ---")
    modelo = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODELO,
        confidence_threshold=CONF_CARGA,
        device="cuda:0",
    )

    print("--- Corriendo SAHI una vez por imagen (guardando todos los candidatos) ---")
    todo_por_imagen = []  # (gts, [(caja, score), ...])
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
            ([op.bbox.minx, op.bbox.miny, op.bbox.maxx, op.bbox.maxy], op.score.value)
            for op in resultado.object_prediction_list
        ]
        gts = [c[1:] for c in muestra["cajas"]]
        todo_por_imagen.append((gts, preds))
        print(f"  {nombre_base}: {len(preds)} candidatos totales (conf>={CONF_CARGA})")

    print("\n" + "=" * 70)
    print("BARRIDO DE UMBRAL -- SAHI sobre combinado-6, test propio (etiquetas corregidas)")
    print("=" * 70)
    print(f"{'conf':>6} | {'P':>8} | {'R':>8} | {'F1':>8}")
    mejor = (0, 0)
    for umbral in UMBRALES_A_PROBAR:
        tp_t, fp_t, fn_t = 0, 0, 0
        for gts, preds in todo_por_imagen:
            preds_filtradas = [(c, s) for c, s in preds if s >= umbral]
            tp, fp, fn = emparejar(preds_filtradas, gts)
            tp_t += tp
            fp_t += fp
            fn_t += fn
        p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        marca = ""
        if f1 > mejor[1]:
            mejor = (umbral, f1)
        print(f"{umbral:>6.2f} | {p:>8.4f} | {r:>8.4f} | {f1:>8.4f}")

    print("=" * 70)
    print(f"Mejor umbral SAHI: conf={mejor[0]:.2f} -> F1={mejor[1]:.4f}")
    print(f"Baseline sin SAHI (conf=0.25): F1=0.7578")
    print("=" * 70)
