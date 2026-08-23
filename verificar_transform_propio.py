"""Verifica si el mismo problema de cajas GT fuera de rango tras la transformacion geometrica
tambien ocurre en propio (no solo en publico) -- critico para saber si el resultado 0.1580 de
Candidato A sobre propio es confiable o tiene el mismo bug."""

import sys
from importlib.machinery import SourceFileLoader

sys.path.insert(0, r"C:\Users\jchag\Documents\TESIS")
ca = SourceFileLoader("candidato_a", r"C:\Users\jchag\Documents\TESIS\Test_all_XML").load_module()

import os
import glob

RAIZ = r"C:\Users\jchag\Documents\TESIS"
dir_images = rf"{RAIZ}\Propio\Tesis.voc\imagess"
dir_annotations = rf"{RAIZ}\Propio\Tesis.voc\annotations"
DIR_SALIDA = rf"{RAIZ}\_tmp_verif"
os.makedirs(DIR_SALIDA, exist_ok=True)

preprocesador = ca.PreprocesamientoMIPE(erosion_borde=0, margen_trampa=0.03, debug=False, dir_salida=DIR_SALIDA)
evaluador = ca.EvaluadorDataset(umbral_iou=0.45)

total_gt, total_fuera = 0, 0
imagenes_con_problema = []

for ruta_xml in sorted(glob.glob(os.path.join(dir_annotations, "*.xml"))):
    nombre_base = os.path.splitext(os.path.basename(ruta_xml))[0]
    ruta_img = os.path.join(dir_images, nombre_base + ".jpg")
    if not os.path.exists(ruta_img):
        continue

    cajas_xml = evaluador.parsear_xml(ruta_xml)
    if not cajas_xml:
        continue

    img_prep, escala, matriz, msg = preprocesador.ejecutar_pipeline(ruta_img)
    if img_prep is None:
        continue

    cajas_gt_t = evaluador.transformar_cajas_gt(cajas_xml, escala, matriz)
    h_w, w_w = img_prep.shape[0], img_prep.shape[1]

    fuera = sum(1 for c in cajas_gt_t if not (0 <= c[0] < w_w and 0 <= c[1] < h_w))
    total_gt += len(cajas_gt_t)
    total_fuera += fuera
    if fuera > 0:
        imagenes_con_problema.append((nombre_base, fuera, len(cajas_gt_t)))

print(f"Total cajas GT (propio): {total_gt}")
print(f"Total fuera de rango tras transformar: {total_fuera} ({100*total_fuera/total_gt:.1f}%)")
print(f"\nImagenes con al menos 1 caja fuera de rango: {len(imagenes_con_problema)}")
for nombre, fuera, total in imagenes_con_problema[:15]:
    print(f"  {nombre}: {fuera}/{total} fuera de rango")
