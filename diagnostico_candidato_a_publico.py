"""Diagnostico de IoU real caja por caja para publico, ya con el fix de EXIF, para confirmar
que el patron (cajas del clasico ~50% mas chicas que el GT) es el mismo que en propio."""

import sys
from importlib.machinery import SourceFileLoader

sys.path.insert(0, r"C:\Users\jchag\Documents\TESIS")
ca = SourceFileLoader("candidato_a_base", r"C:\Users\jchag\Documents\TESIS\Test_all_XML").load_module()

import os
import glob
import cv2

RAIZ = r"C:\Users\jchag\Documents\TESIS"
dir_images = rf"{RAIZ}\md121\images"
dir_annotations = rf"{RAIZ}\md121\annotations"
DIR_SALIDA = rf"{RAIZ}\diagnostico_candidato_a_publico"
os.makedirs(DIR_SALIDA, exist_ok=True)

IMAGENES_A_REVISAR = ["1000", "1281"]

preprocesador = ca.PreprocesamientoMIPE(erosion_borde=0, margen_trampa=0.03, debug=False, dir_salida=DIR_SALIDA)
detector = ca.DetectorUnificado(debug=False, dir_salida=DIR_SALIDA)
evaluador = ca.EvaluadorDataset(umbral_iou=0.45)

# monkeypatch: forzar IMREAD_IGNORE_ORIENTATION dentro de ejecutar_pipeline
_original_imread = cv2.imread
def imread_sin_rotar(ruta, *args, **kwargs):
    return _original_imread(ruta, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
cv2.imread = imread_sin_rotar

for nombre in IMAGENES_A_REVISAR:
    ruta_img = os.path.join(dir_images, f"{nombre}.jpg")
    ruta_xml = os.path.join(dir_annotations, f"{nombre}.xml")
    cajas_xml = evaluador.parsear_xml(ruta_xml)
    img_prep, escala, matriz, msg = preprocesador.ejecutar_pipeline(ruta_img)
    if img_prep is None:
        print(f"{nombre}: descartada ({msg})")
        continue
    cajas_gt = evaluador.transformar_cajas_gt(cajas_xml, escala, matriz)
    img_dibujada, conteo, cajas_pred = detector.detectar_y_dibujar(img_prep.copy())

    print(f"\n=== {nombre} ===  GT={len(cajas_gt)}  Pred={len(cajas_pred)}")
    ious = []
    for i, pred in enumerate(cajas_pred[:8]):
        mejor_iou, mejor_gt = 0, None
        for gt in cajas_gt:
            v = evaluador.calcular_iou(pred, gt)
            if v > mejor_iou:
                mejor_iou, mejor_gt = v, gt
        w_pred, h_pred = pred[2]-pred[0], pred[3]-pred[1]
        if mejor_gt:
            w_gt, h_gt = mejor_gt[2]-mejor_gt[0], mejor_gt[3]-mejor_gt[1]
            print(f"  pred{i}: {w_pred}x{h_pred}px  iou={mejor_iou:.3f}  vs GT {w_gt}x{h_gt}px")
        else:
            print(f"  pred{i}: {w_pred}x{h_pred}px  (sin GT cerca)")

    for pred in cajas_pred:
        mejor_iou = max((evaluador.calcular_iou(pred, gt) for gt in cajas_gt), default=0)
        color = (0, 255, 0) if mejor_iou >= 0.45 else (0, 0, 255)
        cv2.rectangle(img_dibujada, (pred[0], pred[1]), (pred[2], pred[3]), color, 1)
    for gt in cajas_gt:
        cv2.rectangle(img_dibujada, (gt[0], gt[1]), (gt[2], gt[3]), (255, 0, 0), 1)
    cv2.imwrite(os.path.join(DIR_SALIDA, f"{nombre}_iou_detallado.jpg"), img_dibujada)

print(f"\nGuardado en {DIR_SALIDA}")
