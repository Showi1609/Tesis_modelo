"""Verifica si el <width>/<height> declarado en cada XML coincide con las dimensiones reales
del archivo de imagen, para publico y propio -- si estan transpuestos (swap), eso explicaria
por completo el desalineamiento de las cajas GT."""

import glob
import os
import xml.etree.ElementTree as ET

import cv2

def verificar(dir_img, dir_ann, extension_img=".jpg"):
    coinciden, transpuestas, otras = 0, 0, 0
    ejemplos_transpuestas = []
    ejemplos_otras = []
    for ruta_xml in sorted(glob.glob(os.path.join(dir_ann, "*.xml"))):
        nombre_base = os.path.splitext(os.path.basename(ruta_xml))[0]
        ruta_img = os.path.join(dir_img, nombre_base + extension_img)
        if not os.path.exists(ruta_img):
            continue
        try:
            root = ET.parse(ruta_xml).getroot()
            size = root.find("size")
            w_xml = int(size.find("width").text)
            h_xml = int(size.find("height").text)
        except Exception as e:
            continue

        img = cv2.imread(ruta_img)
        if img is None:
            continue
        h_real, w_real = img.shape[:2]

        if w_xml == w_real and h_xml == h_real:
            coinciden += 1
        elif w_xml == h_real and h_xml == w_real:
            transpuestas += 1
            if len(ejemplos_transpuestas) < 5:
                ejemplos_transpuestas.append((nombre_base, w_xml, h_xml, w_real, h_real))
        else:
            otras += 1
            if len(ejemplos_otras) < 5:
                ejemplos_otras.append((nombre_base, w_xml, h_xml, w_real, h_real))

    total = coinciden + transpuestas + otras
    print(f"Total: {total}  Coinciden: {coinciden}  Transpuestas (w/h invertidos): {transpuestas}  Otras discrepancias: {otras}")
    if ejemplos_transpuestas:
        print("Ejemplos transpuestas:")
        for n, wx, hx, wr, hr in ejemplos_transpuestas:
            print(f"  {n}: XML={wx}x{hx}  Real={wr}x{hr}")
    if ejemplos_otras:
        print("Ejemplos con otra discrepancia:")
        for n, wx, hx, wr, hr in ejemplos_otras:
            print(f"  {n}: XML={wx}x{hx}  Real={wr}x{hr}")


print("=== PUBLICO (md121) ===")
verificar(r"C:\Users\jchag\Documents\TESIS\md121\images", r"C:\Users\jchag\Documents\TESIS\md121\annotations")

print("\n=== PROPIO ===")
verificar(r"C:\Users\jchag\Documents\TESIS\Propio\Tesis.voc\imagess", r"C:\Users\jchag\Documents\TESIS\Propio\Tesis.voc\annotations")
