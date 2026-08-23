"""
Aplica al XML original (Propio\\Tesis.voc\\annotations\\) las cajas confirmadas durante la
revision manual de falsos positivos (Etapa 15): solo los recortes que (a) siguen existiendo en
revision_fp_test_propio\\ (el usuario borro los que NO eran mosca real) y (b) tienen
confianza >= UMBRAL_MIN (0.35 -- filtro acordado para descartar los mas ambiguos).

Ojo con la escala: las coordenadas del manifest estan en el espacio de la imagen usada para
inferencia (yolo_dataset_desglose, redimensionada a max 1920px), pero el XML original referencia
la imagen a resolucion completa (hasta 4000+px). Se reescala por imagen antes de escribir.
"""

import csv
import os

from PIL import Image

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DIR_REVISION = rf"{RAIZ}\revision_fp_test_propio"
MANIFEST = rf"{DIR_REVISION}\manifest_fp.csv"
DIR_IMG_DESGLOSE = rf"{RAIZ}\yolo_dataset_desglose\images\test_propio"
DIR_IMG_ORIGINAL = rf"{RAIZ}\Propio\Tesis.voc\imagess"
DIR_ANOTACIONES = rf"{RAIZ}\Propio\Tesis.voc\annotations"
UMBRAL_MIN = 0.35


def nombre_xml_original(imagen_origen):
    """'propio_2012_jpg.rf.XXX.jpg' -> '2012_jpg.rf.XXX.xml' (quita el prefijo 'propio_' que
    agrega yolo_entrenamiento.py, y el que usa el XML original no lo tiene)."""
    base = os.path.splitext(imagen_origen)[0]
    if base.startswith("propio_"):
        base = base[len("propio_"):]
    return base + ".xml", base + ".jpg"


def bloque_object_xml(xmin, ymin, xmax, ymax):
    return (
        "\t<object>\n"
        "\t\t<name>WF</name>\n"
        "\t\t<pose>Unspecified</pose>\n"
        "\t\t<truncated>0</truncated>\n"
        "\t\t<difficult>0</difficult>\n"
        "\t\t<occluded>0</occluded>\n"
        "\t\t<bndbox>\n"
        f"\t\t\t<xmin>{xmin}</xmin>\n"
        f"\t\t\t<xmax>{xmax}</xmax>\n"
        f"\t\t\t<ymin>{ymin}</ymin>\n"
        f"\t\t\t<ymax>{ymax}</ymax>\n"
        "\t\t</bndbox>\n"
        "\t</object>\n"
    )


if __name__ == "__main__":
    with open(MANIFEST) as f:
        manifest = list(csv.DictReader(f))

    confirmados = os.listdir(DIR_REVISION)
    confirmados = set(c for c in confirmados if c.endswith(".jpg"))

    filas_confirmadas = [
        m for m in manifest
        if m["crop"] in confirmados and float(m["confianza"]) >= UMBRAL_MIN
    ]
    print(f"{len(filas_confirmadas)} correcciones confirmadas (existentes + conf>={UMBRAL_MIN})")

    por_imagen = {}
    for fila in filas_confirmadas:
        por_imagen.setdefault(fila["imagen_origen"], []).append(fila)

    total_agregadas = 0
    for imagen_origen, filas in por_imagen.items():
        nombre_xml, nombre_img_original = nombre_xml_original(imagen_origen)
        ruta_xml = os.path.join(DIR_ANOTACIONES, nombre_xml)
        ruta_img_desglose = os.path.join(DIR_IMG_DESGLOSE, imagen_origen)
        ruta_img_original = os.path.join(DIR_IMG_ORIGINAL, nombre_img_original)

        if not os.path.exists(ruta_xml):
            print(f"  [!] No existe {ruta_xml}, se salta")
            continue

        with Image.open(ruta_img_desglose) as img_d:
            w_d, h_d = img_d.size
        with Image.open(ruta_img_original) as img_o:
            w_o, h_o = img_o.size
        escala_x, escala_y = w_o / w_d, h_o / h_d
        print(f"\n{imagen_origen}: desglose {w_d}x{h_d} -> original {w_o}x{h_o} "
              f"(escala {escala_x:.3f}x, {escala_y:.3f}y), {len(filas)} cajas nuevas")

        bloques = ""
        for fila in filas:
            x1 = float(fila["x1"]) * escala_x
            y1 = float(fila["y1"]) * escala_y
            x2 = float(fila["x2"]) * escala_x
            y2 = float(fila["y2"]) * escala_y
            bloques += bloque_object_xml(round(x1), round(y1), round(x2), round(y2))

        with open(ruta_xml, encoding="utf-8") as f:
            contenido = f.read()

        marcador = "<metadata>" if "<metadata>" in contenido else "</annotation>"
        idx = contenido.rfind(marcador)
        nuevo_contenido = contenido[:idx] + bloques + contenido[idx:]

        with open(ruta_xml, "w", encoding="utf-8") as f:
            f.write(nuevo_contenido)

        total_agregadas += len(filas)

    print(f"\n=== {total_agregadas} cajas agregadas en {len(por_imagen)} archivos XML ===")
