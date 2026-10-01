"""Auditoria (2026-09-12): recalcula el 5-fold de combinado6_v2 usando emparejamiento manual
explicito (evaluar_con_match de evaluar_combinado6_v2_iou_match.py) en vez de modelo.val() de
Ultralytics, para que sea comparable con el mismo criterio que el Candidato A. No modifica
kfold_cv_v2.py ni evaluar_combinado6_v2_iou_match.py -- los reutiliza via import.

Ademas guarda resultados POR IMAGEN (no solo por fold) para poder aislar el F1 de las 5
imagenes de salto masivo (2024, 2079, 2080, 2081, 2082) frente a las 55 restantes."""

import glob
import json
import os

import numpy as np
from ultralytics import YOLO

from evaluar_combinado6_v2_iou_match import iou as iou_xyxy
from evaluar_combinado6_v2_iou_match import leer_gt_yolo

RAIZ = r"C:\Users\jchag\Documents\TESIS"
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45
UMBRALES = (0.1, 0.45)
SALTO_MASIVO = {"2024", "2079", "2080", "2081", "2082"}  # numero base de la imagen propia


def evaluar_imagen(modelo, ruta_img, ruta_lbl, iou_match):
    resultado = modelo.predict(source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS,
                                device=0, verbose=False)[0]
    img_h, img_w = resultado.orig_shape
    preds = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in resultado.boxes]
    gts = leer_gt_yolo(ruta_lbl, img_w, img_h)

    preds_ordenadas = sorted(preds, key=lambda p: -p[1])
    gt_usados = [False] * len(gts)
    tp = 0
    for caja_pred, _ in preds_ordenadas:
        mejor_iou, mejor_j = 0, -1
        for j, gt in enumerate(gts):
            if gt_usados[j]:
                continue
            v = iou_xyxy(caja_pred, gt)
            if v > mejor_iou:
                mejor_iou, mejor_j = v, j
        if mejor_iou >= iou_match:
            gt_usados[mejor_j] = True
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn, len(gts)


def numero_base(nombre_archivo):
    """'propio_2024_jpg.rf.XXXX.jpg' -> '2024'"""
    base = nombre_archivo.replace("propio_", "", 1)
    return base.split("_")[0]


def f1_de(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


if __name__ == "__main__":
    resultados_por_fold = {u: [] for u in UMBRALES}
    resultados_por_imagen = []  # lista de dicts: fold, imagen, numero_base, tp/fp/fn por umbral

    for fold_idx in range(1, 6):
        nombre_fold = f"fold{fold_idx}"
        modelo_path = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_{nombre_fold}_v2\weights\best.pt"
        dir_img = rf"{RAIZ}\yolo_dataset_{nombre_fold}_test_propio_v2\images\train"
        dir_lbl = rf"{RAIZ}\yolo_dataset_{nombre_fold}_test_propio_v2\labels\train"

        print(f"\n{'='*60}\n{nombre_fold}\n{'='*60}")
        modelo = YOLO(modelo_path)
        rutas_img = sorted(glob.glob(os.path.join(dir_img, "*.jpg")))

        acumulado = {u: {"tp": 0, "fp": 0, "fn": 0} for u in UMBRALES}
        for ruta_img in rutas_img:
            nombre = os.path.basename(ruta_img)
            nombre_base_archivo = os.path.splitext(nombre)[0]
            ruta_lbl = os.path.join(dir_lbl, nombre_base_archivo + ".txt")
            num_base = numero_base(nombre_base_archivo)

            fila = {"fold": nombre_fold, "imagen": nombre_base_archivo, "numero_base": num_base}
            for u in UMBRALES:
                tp, fp, fn, n_real = evaluar_imagen(modelo, ruta_img, ruta_lbl, u)
                acumulado[u]["tp"] += tp
                acumulado[u]["fp"] += fp
                acumulado[u]["fn"] += fn
                fila[f"tp_{u}"], fila[f"fp_{u}"], fila[f"fn_{u}"] = tp, fp, fn
            fila["n_real"] = n_real
            resultados_por_imagen.append(fila)
            print(f"  {nombre_base_archivo} (base={num_base}): "
                  f"iou0.1 tp{fila['tp_0.1']}/fp{fila['fp_0.1']}/fn{fila['fn_0.1']}  "
                  f"iou0.45 tp{fila['tp_0.45']}/fp{fila['fp_0.45']}/fn{fila['fn_0.45']}")

        for u in UMBRALES:
            f1 = f1_de(acumulado[u]["tp"], acumulado[u]["fp"], acumulado[u]["fn"])
            resultados_por_fold[u].append(f1)
            print(f"  -- {nombre_fold} iou={u}: F1={f1:.4f} "
                  f"(TP={acumulado[u]['tp']} FP={acumulado[u]['fp']} FN={acumulado[u]['fn']})")

    print(f"\n{'='*60}\nRESULTADO FINAL -- 5-FOLD CON MATCHING MANUAL\n{'='*60}")
    resumen = {}
    for u in UMBRALES:
        f1s = resultados_por_fold[u]
        media, std = float(np.mean(f1s)), float(np.std(f1s))
        resumen[u] = {"f1_por_fold": f1s, "media": media, "std": std}
        print(f"iou={u}: F1 por fold = {[round(f, 4) for f in f1s]}")
        print(f"iou={u}: media={media:.4f}  std={std:.4f}")

    with open(rf"{RAIZ}\kfold_iou_match_resultados.json", "w") as f:
        json.dump({"resumen": resumen, "por_imagen": resultados_por_imagen}, f, indent=2)
    print(f"\nGuardado: kfold_iou_match_resultados.json")
