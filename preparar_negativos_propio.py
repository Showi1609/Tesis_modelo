"""
Etapa 17: prepara las 24 fotos propias sin mosca blanca (trips, pulgones, otros bichos en la
trampa amarilla) como "negativos duros" para train -- fondos reales del invernadero + insectos
reales que NO son WF, con anotacion vacia (0 cajas). Nunca entran a val/test, solo a train.
"""

import glob
import os

from PIL import Image

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DIR_ORIGEN = rf"{RAIZ}\Propio\Tesis.voc\Propio"
DIR_SALIDA_IMG = rf"{RAIZ}\Propio_negativos\images"
DIR_SALIDA_ANN = rf"{RAIZ}\Propio_negativos\annotations"

# los 24 numeros confirmados como "sin mosca blanca" (nunca anotados en Roboflow)
NUMEROS_SIN_WF = [
    2000, 2001, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2025, 2026, 2027, 2028,
    2031, 2032, 2033, 2036, 2038, 2039, 2057, 2072, 2076, 2077, 2083,
]


def escribir_xml_vacio(ruta_xml, nombre_archivo, w, h):
    contenido = (
        "<annotation>\n"
        "\t<folder></folder>\n"
        f"\t<filename>{nombre_archivo}</filename>\n"
        f"\t<path>{nombre_archivo}</path>\n"
        "\t<source>\n"
        "\t\t<database>propio_negativos</database>\n"
        "\t</source>\n"
        "\t<size>\n"
        f"\t\t<width>{w}</width>\n"
        f"\t\t<height>{h}</height>\n"
        "\t\t<depth>3</depth>\n"
        "\t</size>\n"
        "\t<segmented>0</segmented>\n"
        "</annotation>\n"
    )
    with open(ruta_xml, "w", encoding="utf-8") as f:
        f.write(contenido)


if __name__ == "__main__":
    os.makedirs(DIR_SALIDA_IMG, exist_ok=True)
    os.makedirs(DIR_SALIDA_ANN, exist_ok=True)

    total = 0
    for numero in NUMEROS_SIN_WF:
        ruta_origen = os.path.join(DIR_ORIGEN, f"{numero}.jpg")
        if not os.path.exists(ruta_origen):
            print(f"  [!] No existe {ruta_origen}, se salta")
            continue

        nombre_archivo = f"neg_{numero}.jpg"
        ruta_img_salida = os.path.join(DIR_SALIDA_IMG, nombre_archivo)
        with Image.open(ruta_origen) as img:
            img = img.convert("RGB")
            w, h = img.size
            img.save(ruta_img_salida, "JPEG", quality=95)

        ruta_xml_salida = os.path.join(DIR_SALIDA_ANN, f"neg_{numero}.xml")
        escribir_xml_vacio(ruta_xml_salida, nombre_archivo, w, h)
        total += 1
        print(f"  {numero}.jpg -> {nombre_archivo} ({w}x{h}, 0 cajas)")

    print(f"\n=== {total} negativos preparados en {DIR_SALIDA_IMG} y {DIR_SALIDA_ANN} ===")
