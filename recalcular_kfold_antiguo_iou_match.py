"""Auditoria (2026-09-12): reconstruye la serie 'ANTES' (Etapa 18, modelo y etiquetas viejas)
del 5-fold con el MISMO metodo de matching manual usado para 'DESPUES' (Etapa 19b, ver
recalcular_kfold_iou_match.py), para que la Figura 30 sea comparable punto a punto.

Reutiliza exactamente los mismos 12 imagenes por fold que ya uso 'despues' (misma semilla=42,
mismo split -- confirmado leyendo kfold_cv.py: SEMILLA=42, N_FOLDS=5, misma logica de shuffle
que kfold_cv_v2.py). El modelo viejo es yolo_runs/whitefly_yolov8n_fold{i}/weights/best.pt
(sin _v2). El ground truth viejo se toma de Propio/Tesis.voc/annotations_backup_20260821/
(backup pre-Etapa-19), escalado a la resolucion de la imagen de fold ya redimensionada.

No modifica ningun script existente."""

import glob
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DIR_BACKUP_ANN = rf"{RAIZ}\Propio\Tesis.voc\annotations_backup_20260821"
DIR_IMG_ORIGINAL = rf"{RAIZ}\Propio\Tesis.voc\imagess"
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45
UMBRALES = (0.1, 0.45)


def parsear_xml_backup(ruta_xml):
    cajas = []
    try:
        root = ET.parse(ruta_xml).getroot()
        for obj in root.findall("object"):
            if obj.find("name").text != "WF":
                continue
            b = obj.find("bndbox")
            cajas.append([float(b.find("xmin").text), float(b.find("ymin").text),
                          float(b.find("xmax").text), float(b.find("ymax").text)])
    except Exception as e:
        print(f"Error leyendo {ruta_xml}: {e}")
    return cajas


def numero_base_y_xml(nombre_archivo_fold):
    """'propio_2079_jpg.rf.XXX.jpg' -> ('2079', '2079_jpg.rf.XXX.xml', '2079_jpg.rf.XXX.jpg')"""
    base = nombre_archivo_fold.replace("propio_", "", 1)
    numero = base.split("_")[0]
    return numero, base + ".xml", base + ".jpg"


def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def cargar_gt_viejo_escalado(ruta_img_fold, nombre_xml, nombre_img_original):
    ruta_xml = os.path.join(DIR_BACKUP_ANN, nombre_xml)
    if not os.path.exists(ruta_xml):
        print(f"    [!] No existe backup XML para {nombre_xml}")
        return []
    cajas_orig = parsear_xml_backup(ruta_xml)
    if not cajas_orig:
        return []

    with Image.open(ruta_img_fold) as img_f:
        w_f, h_f = img_f.size
    ruta_img_o = os.path.join(DIR_IMG_ORIGINAL, nombre_img_original)
    with Image.open(ruta_img_o) as img_o:
        w_o, h_o = img_o.size
    esc_x, esc_y = w_f / w_o, h_f / h_o

    return [[x1 * esc_x, y1 * esc_y, x2 * esc_x, y2 * esc_y] for x1, y1, x2, y2 in cajas_orig]


def evaluar_imagen(modelo, ruta_img, gts, iou_match):
    resultado = modelo.predict(source=ruta_img, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS,
                                device=0, verbose=False)[0]
    preds = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in resultado.boxes]
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
    return tp, fp, fn


def f1_de(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


if __name__ == "__main__":
    resultados_por_fold = {u: [] for u in UMBRALES}
    resultados_por_imagen = []

    for fold_idx in range(1, 6):
        nombre_fold = f"fold{fold_idx}"
        modelo_path = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_{nombre_fold}\weights\best.pt"  # SIN _v2
        dir_img = rf"{RAIZ}\yolo_dataset_{nombre_fold}_test_propio_v2\images\train"  # mismo split de imagenes

        if not os.path.exists(modelo_path):
            print(f"[!] No existe {modelo_path}, se salta {nombre_fold}")
            continue

        print(f"\n{'='*60}\n{nombre_fold} (modelo Etapa 18, etiquetas backup pre-Etapa-19)\n{'='*60}")
        modelo = YOLO(modelo_path)
        rutas_img = sorted(glob.glob(os.path.join(dir_img, "*.jpg")))

        acumulado = {u: {"tp": 0, "fp": 0, "fn": 0} for u in UMBRALES}
        for ruta_img in rutas_img:
            nombre_archivo = os.path.splitext(os.path.basename(ruta_img))[0]
            numero, nombre_xml, nombre_img_original = numero_base_y_xml(nombre_archivo)
            gts = cargar_gt_viejo_escalado(ruta_img, nombre_xml, nombre_img_original)

            fila = {"fold": nombre_fold, "imagen": nombre_archivo, "numero_base": numero, "n_real_viejo": len(gts)}
            for u in UMBRALES:
                tp, fp, fn = evaluar_imagen(modelo, ruta_img, gts, u)
                acumulado[u]["tp"] += tp
                acumulado[u]["fp"] += fp
                acumulado[u]["fn"] += fn
                fila[f"tp_{u}"], fila[f"fp_{u}"], fila[f"fn_{u}"] = tp, fp, fn
            resultados_por_imagen.append(fila)
            print(f"  {nombre_archivo} (base={numero}, GT_viejo={len(gts)}): "
                  f"iou0.1 tp{fila['tp_0.1']}/fp{fila['fp_0.1']}/fn{fila['fn_0.1']}  "
                  f"iou0.45 tp{fila['tp_0.45']}/fp{fila['fp_0.45']}/fn{fila['fn_0.45']}")

        for u in UMBRALES:
            f1 = f1_de(acumulado[u]["tp"], acumulado[u]["fp"], acumulado[u]["fn"])
            resultados_por_fold[u].append(f1)
            print(f"  -- {nombre_fold} iou={u}: F1={f1:.4f} "
                  f"(TP={acumulado[u]['tp']} FP={acumulado[u]['fp']} FN={acumulado[u]['fn']})")

    print(f"\n{'='*60}\nRESULTADO -- 5-FOLD 'ANTES' (Etapa 18) CON MATCHING MANUAL\n{'='*60}")
    resumen = {}
    for u in UMBRALES:
        f1s = resultados_por_fold[u]
        media, std = float(np.mean(f1s)), float(np.std(f1s))
        resumen[u] = {"f1_por_fold": f1s, "media": media, "std": std}
        print(f"iou={u}: F1 por fold = {[round(f, 4) for f in f1s]}")
        print(f"iou={u}: media={media:.4f}  std={std:.4f}")

    with open(rf"{RAIZ}\kfold_antiguo_iou_match_resultados.json", "w") as f:
        json.dump({"resumen": resumen, "por_imagen": resultados_por_imagen}, f, indent=2)
    print(f"\nGuardado: kfold_antiguo_iou_match_resultados.json")
