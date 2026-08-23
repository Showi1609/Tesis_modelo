"""Diagnostico: para unas pocas imagenes, calcula el IoU real entre cada prediccion del
candidato A y su mejor match de GT, y recorta la zona para inspeccionar visualmente si las
cajas estan realmente cerca (blob impreciso) o hay algo mas roto (offset sistematico, etc.)."""

import os
import sys
from importlib.machinery import SourceFileLoader

sys.path.insert(0, r"C:\Users\jchag\Documents\TESIS")
ca = SourceFileLoader("candidato_a", r"C:\Users\jchag\Documents\TESIS\Test_all_XML").load_module()

import cv2

RAIZ = r"C:\Users\jchag\Documents\TESIS"
dir_base = rf"{RAIZ}\Propio\Tesis.voc"
dir_images = os.path.join(dir_base, "imagess")
dir_annotations = os.path.join(dir_base, "annotations")
DIR_SALIDA = rf"{RAIZ}\diagnostico_candidato_a"
os.makedirs(DIR_SALIDA, exist_ok=True)

IMAGENES_A_REVISAR = ["2007", "2013", "2044"]

preprocesador = ca.PreprocesamientoMIPE(erosion_borde=0, margen_trampa=0.03, debug=False, dir_salida=DIR_SALIDA)
detector = ca.DetectorUnificado(debug=False, dir_salida=DIR_SALIDA)
evaluador = ca.EvaluadorDataset(umbral_iou=0.45)

for prefijo in IMAGENES_A_REVISAR:
    import glob
    candidatos = glob.glob(os.path.join(dir_images, f"{prefijo}_jpg.rf.*.jpg"))
    if not candidatos:
        continue
    ruta_img = candidatos[0]
    nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
    ruta_xml = os.path.join(dir_annotations, f"{nombre_base}.xml")

    cajas_xml = evaluador.parsear_xml(ruta_xml)
    img_prep, escala, matriz, msg = preprocesador.ejecutar_pipeline(ruta_img)
    if img_prep is None:
        print(f"{nombre_base}: descartada ({msg})")
        continue
    cajas_gt = evaluador.transformar_cajas_gt(cajas_xml, escala, matriz)
    img_dibujada, conteo, cajas_pred = detector.detectar_y_dibujar(img_prep.copy())

    print(f"\n=== {nombre_base} ===  GT={len(cajas_gt)}  Pred={len(cajas_pred)}")
    for i, pred in enumerate(cajas_pred):
        mejor_iou, mejor_gt = 0, None
        for gt in cajas_gt:
            v = evaluador.calcular_iou(pred, gt)
            if v > mejor_iou:
                mejor_iou, mejor_gt = v, gt
        w_pred = pred[2] - pred[0]
        h_pred = pred[3] - pred[1]
        w_gt = (mejor_gt[2] - mejor_gt[0]) if mejor_gt else 0
        h_gt = (mejor_gt[3] - mejor_gt[1]) if mejor_gt else 0
        print(f"  pred{i}: caja={pred} ({w_pred}x{h_pred}px)  mejor_iou={mejor_iou:.3f}"
              + (f"  vs GT={mejor_gt} ({w_gt}x{h_gt}px)" if mejor_gt else "  (sin GT cerca)"))

    # marcar el IoU sobre la imagen para cada prediccion
    for pred in cajas_pred:
        mejor_iou = max((evaluador.calcular_iou(pred, gt) for gt in cajas_gt), default=0)
        color = (0, 255, 0) if mejor_iou >= 0.45 else (0, 0, 255)
        cv2.rectangle(img_dibujada, (pred[0], pred[1]), (pred[2], pred[3]), color, 1)
        cv2.putText(img_dibujada, f"{mejor_iou:.2f}", (pred[0], pred[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    for gt in cajas_gt:
        cv2.rectangle(img_dibujada, (gt[0], gt[1]), (gt[2], gt[3]), (255, 0, 0), 1)

    cv2.imwrite(os.path.join(DIR_SALIDA, f"{nombre_base}_iou_detallado.jpg"), img_dibujada)

print(f"\nGuardado en {DIR_SALIDA}")
