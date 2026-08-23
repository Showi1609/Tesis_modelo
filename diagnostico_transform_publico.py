"""Diagnostico especifico: revisa paso a paso la transformacion de las cajas GT (XML original)
al espacio de la imagen ya corregida geometricamente, para una imagen publica -- para confirmar
si el problema es el detector (ruido real) o la transformacion de coordenadas GT (bug)."""

import sys
from importlib.machinery import SourceFileLoader

sys.path.insert(0, r"C:\Users\jchag\Documents\TESIS")
ca = SourceFileLoader("candidato_a", r"C:\Users\jchag\Documents\TESIS\Test_all_XML").load_module()

import os
import cv2
import numpy as np

RAIZ = r"C:\Users\jchag\Documents\TESIS"
dir_images = rf"{RAIZ}\md121\images"
dir_annotations = rf"{RAIZ}\md121\annotations"
DIR_SALIDA = rf"{RAIZ}\diagnostico_transform_publico"
os.makedirs(DIR_SALIDA, exist_ok=True)

NOMBRE = "1000"

preprocesador = ca.PreprocesamientoMIPE(erosion_borde=0, margen_trampa=0.03, debug=False, dir_salida=DIR_SALIDA)
detector = ca.DetectorUnificado(debug=False, dir_salida=DIR_SALIDA)
evaluador = ca.EvaluadorDataset(umbral_iou=0.45)

ruta_xml = os.path.join(dir_annotations, f"{NOMBRE}.xml")
ruta_img = os.path.join(dir_images, f"{NOMBRE}.jpg")

cajas_xml = evaluador.parsear_xml(ruta_xml)
print(f"Cajas GT originales (espacio XML, sin transformar): {len(cajas_xml)}")
for c in cajas_xml[:5]:
    print(f"  {c}")

img_original = cv2.imread(ruta_img)
h_o, w_o = img_original.shape[:2]
print(f"\nDimension original de la imagen: {w_o}x{h_o}")

img_prep, escala, matriz, msg = preprocesador.ejecutar_pipeline(ruta_img)
print(f"\nescala aplicada (resize inicial): {escala}")
print(f"matriz de perspectiva es None? {matriz is None}")
if img_prep is not None:
    print(f"Dimension de la imagen ya corregida (warp): {img_prep.shape[1]}x{img_prep.shape[0]}")

cajas_gt_transformadas = evaluador.transformar_cajas_gt(cajas_xml, escala, matriz)
print(f"\nCajas GT DESPUES de transformar (deberian caer dentro de {img_prep.shape[1]}x{img_prep.shape[0]}):")
fuera_de_rango = 0
for c in cajas_gt_transformadas[:8]:
    dentro = (0 <= c[0] < img_prep.shape[1]) and (0 <= c[1] < img_prep.shape[0])
    if not dentro:
        fuera_de_rango += 1
    print(f"  {c}  {'DENTRO' if dentro else '*** FUERA DE RANGO ***'}")

for c in cajas_gt_transformadas:
    dentro = (0 <= c[0] < img_prep.shape[1]) and (0 <= c[1] < img_prep.shape[0])
    if not dentro:
        fuera_de_rango += 1
print(f"\nTotal cajas GT transformadas fuera de rango de la imagen warpeada: {fuera_de_rango}/{len(cajas_gt_transformadas)}")

# dibujar todo para inspeccion visual
img_dibujada, conteo, cajas_pred = detector.detectar_y_dibujar(img_prep.copy())
for gt in cajas_gt_transformadas:
    cv2.rectangle(img_dibujada, (gt[0], gt[1]), (gt[2], gt[3]), (255, 0, 0), 2)
for pred in cajas_pred:
    cv2.rectangle(img_dibujada, (pred[0], pred[1]), (pred[2], pred[3]), (0, 255, 0), 1)
cv2.imwrite(os.path.join(DIR_SALIDA, f"{NOMBRE}_diagnostico.jpg"), img_dibujada)
print(f"\nGuardado: {DIR_SALIDA}\\{NOMBRE}_diagnostico.jpg")
