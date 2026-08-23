"""
Etapa 13 (whitefly_yolov8n_combinado-9): copy-paste con recortes MEZCLADOS -- publico (md121) Y
propio a la vez, en el mismo generador. Combina lo que funciono de cada intento anterior:

  - Etapa 6 (combinado-6, recortes 100% publicos): mejora real, pero la mosca pegada trae su
    propio mismatch de dominio (otra camara/luz) aunque el fondo sea propio.
  - Etapa 12 (combinado-8, recortes 100% propios): visualmente mas consistente, pero el pool de
    solo 42 fotos propias aporta poca variedad nueva -- resultado sin mejora (F1 propio bajo).

Aqui cada mosca pegada se sortea 50/50 entre el pool publico (5807 recortes, mucha variedad) y
el pool propio (357 recortes tras el tope de 18/imagen, visualmente consistente) -- la intencion
es quedarse con la variedad real del publico sin perder toda la consistencia del propio.

Densidad identica a combinado-6 (3-8 moscas/variante) para no mezclar mas de una variable nueva
a la vez.
"""

import os
import sys
import glob
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, r"C:\Users\jchag\Documents\TESIS")
import yolo_entrenamiento as yt

DIR_SALIDA = r"C:\Users\jchag\Documents\TESIS\Propio_copypaste_v4"
VARIANTES_POR_IMAGEN = 3
MOSCAS_POR_VARIANTE_MIN = 3
MOSCAS_POR_VARIANTE_MAX = 8
MAX_RECORTES_POR_IMAGEN_PROPIO = 18   # igual que Etapa 12, evita que la imagen de 271 cajas domine
PROB_USAR_PROPIO = 0.5                # 50/50 entre pool publico y pool propio en cada mosca pegada
PADDING_RECORTE = 0.15
MAX_INTENTOS_UBICACION = 30
SOLAPAMIENTO_MAX = 0.05
SEMILLA = 123

random.seed(SEMILLA)


def cargar_recortes_publico():
    fuente_publico = next(f for f in yt.FUENTES if f["grupo"] == "publico")
    recortes = []
    rutas_xml = glob.glob(os.path.join(fuente_publico["dir_anotaciones"], "*.xml"))
    for ruta_xml in rutas_xml:
        nombre_base = os.path.splitext(os.path.basename(ruta_xml))[0]
        ruta_img = os.path.join(fuente_publico["dir_imagenes"], nombre_base + ".jpg")
        if not os.path.exists(ruta_img):
            continue
        cajas = yt.parsear_voc(ruta_xml, yt.CLASES)
        if not cajas:
            continue
        try:
            with Image.open(ruta_img) as img:
                img = img.convert("RGB")
                for _, xmin, ymin, xmax, ymax in cajas:
                    w, h = xmax - xmin, ymax - ymin
                    if w <= 0 or h <= 0:
                        continue
                    pad_x, pad_y = w * PADDING_RECORTE, h * PADDING_RECORTE
                    x0 = max(0, int(xmin - pad_x))
                    y0 = max(0, int(ymin - pad_y))
                    x1 = min(img.width, int(xmax + pad_x))
                    y1 = min(img.height, int(ymax + pad_y))
                    if x1 - x0 < 6 or y1 - y0 < 6:
                        continue
                    recortes.append(img.crop((x0, y0, x1, y1)).copy())
        except Exception as e:
            print(f"Error leyendo {ruta_img}: {e}")
    return recortes


def cargar_recortes_propio_train(train_propio):
    recortes = []
    for muestra in train_propio:
        cajas = muestra["cajas"]
        if not cajas:
            continue
        indices = list(range(len(cajas)))
        random.shuffle(indices)
        indices = indices[:MAX_RECORTES_POR_IMAGEN_PROPIO]
        try:
            with Image.open(muestra["ruta_imagen"]) as img:
                img = img.convert("RGB")
                for i in indices:
                    _, xmin, ymin, xmax, ymax = cajas[i]
                    w, h = xmax - xmin, ymax - ymin
                    if w <= 0 or h <= 0:
                        continue
                    pad_x, pad_y = w * PADDING_RECORTE, h * PADDING_RECORTE
                    x0 = max(0, int(xmin - pad_x))
                    y0 = max(0, int(ymin - pad_y))
                    x1 = min(img.width, int(xmax + pad_x))
                    y1 = min(img.height, int(ymax + pad_y))
                    if x1 - x0 < 6 or y1 - y0 < 6:
                        continue
                    recortes.append(img.crop((x0, y0, x1, y1)).copy())
        except Exception as e:
            print(f"Error leyendo {muestra['ruta_imagen']}: {e}")
    return recortes


def mascara_elipse_suave(w, h):
    mascara = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mascara)
    margen = max(1, int(min(w, h) * 0.08))
    draw.ellipse((margen, margen, w - margen, h - margen), fill=255)
    radio_blur = max(1, int(min(w, h) * 0.12))
    return mascara.filter(ImageFilter.GaussianBlur(radio_blur))


def iou(caja_a, caja_b):
    xa, ya = max(caja_a[0], caja_b[0]), max(caja_a[1], caja_b[1])
    xb, yb = min(caja_a[2], caja_b[2]), min(caja_a[3], caja_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (caja_a[2] - caja_a[0]) * (caja_a[3] - caja_a[1])
    area_b = (caja_b[2] - caja_b[0]) * (caja_b[3] - caja_b[1])
    return inter / float(area_a + area_b - inter)


def detectar_area_trampa(imagen_pil):
    arr = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    amarillo_bajo = np.array([15, 80, 80])
    amarillo_alto = np.array([32, 255, 255])
    mascara = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)
    kernel = np.ones((15, 15), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    cnt = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(cnt) < (imagen_pil.width * imagen_pil.height * 0.05):
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    margen = int(min(w, h) * 0.04)
    return [x + margen, y + margen, x + w - margen, y + h - margen]


def encontrar_ubicacion(cajas_existentes, w_img, h_img, w_crop, h_crop, area_trampa):
    if area_trampa is not None:
        tx0, ty0, tx1, ty1 = area_trampa
        x_min, x_max = tx0, max(tx0, tx1 - w_crop)
        y_min, y_max = ty0, max(ty0, ty1 - h_crop)
        if x_max <= x_min or y_max <= y_min:
            return None
    else:
        x_min, x_max = 0, max(0, w_img - w_crop)
        y_min, y_max = 0, max(0, h_img - h_crop)

    for _ in range(MAX_INTENTOS_UBICACION):
        x0 = random.randint(x_min, x_max)
        y0 = random.randint(y_min, y_max)
        candidata = [x0, y0, x0 + w_crop, y0 + h_crop]
        if all(iou(candidata, c) <= SOLAPAMIENTO_MAX for c in cajas_existentes):
            return x0, y0
    return None


def ajustar_brillo(recorte, fondo, x0, y0):
    w, h = recorte.size
    parche = fondo.crop((x0, y0, x0 + w, y0 + h)).convert("L")
    brillo_fondo = np.array(parche, dtype=np.float32).mean()
    brillo_recorte = np.array(recorte.convert("L"), dtype=np.float32).mean()
    if brillo_recorte < 1:
        return recorte
    factor = np.clip(brillo_fondo / brillo_recorte, 0.6, 1.6)
    arr = np.clip(np.array(recorte, dtype=np.float32) * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def area_desde_cajas(cajas_originales, w_img, h_img, expansion=0.15):
    if not cajas_originales:
        return None
    xs0 = [c[1] for c in cajas_originales]
    ys0 = [c[2] for c in cajas_originales]
    xs1 = [c[3] for c in cajas_originales]
    ys1 = [c[4] for c in cajas_originales]
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    w, h = x1 - x0, y1 - y0
    return [
        max(0, x0 - w * expansion), max(0, y0 - h * expansion),
        min(w_img, x1 + w * expansion), min(h_img, y1 + h * expansion),
    ]


def generar_variante(ruta_img_fondo, cajas_originales, recortes_publico, recortes_propio):
    with Image.open(ruta_img_fondo) as fondo:
        fondo = fondo.convert("RGB").copy()
    w_img, h_img = fondo.size
    cajas_actuales = [list(c[1:]) for c in cajas_originales]
    nuevas_cajas = []

    area_trampa = detectar_area_trampa(fondo)
    if area_trampa is None:
        area_trampa = area_desde_cajas(cajas_originales, w_img, h_img)

    n_moscas = random.randint(MOSCAS_POR_VARIANTE_MIN, MOSCAS_POR_VARIANTE_MAX)
    for _ in range(n_moscas):
        pool = recortes_propio if random.random() < PROB_USAR_PROPIO else recortes_publico
        recorte = random.choice(pool)
        w_crop, h_crop = recorte.size
        if w_crop >= w_img or h_crop >= h_img:
            continue
        ubicacion = encontrar_ubicacion(cajas_actuales, w_img, h_img, w_crop, h_crop, area_trampa)
        if ubicacion is None:
            continue
        x0, y0 = ubicacion
        recorte_ajustado = ajustar_brillo(recorte, fondo, x0, y0)
        mascara = mascara_elipse_suave(w_crop, h_crop)
        fondo.paste(recorte_ajustado, (x0, y0), mascara)
        caja_nueva = [x0 + w_crop * 0.15, y0 + h_crop * 0.15, x0 + w_crop * 0.85, y0 + h_crop * 0.85]
        cajas_actuales.append(caja_nueva)
        nuevas_cajas.append(caja_nueva)

    todas_las_cajas = [c[1:] for c in cajas_originales] + nuevas_cajas
    return fondo, todas_las_cajas


def escribir_voc_xml(ruta_xml, nombre_archivo, w, h, cajas):
    lineas = [
        "<annotation>",
        f"  <filename>{nombre_archivo}</filename>",
        "  <size>",
        f"    <width>{w}</width>",
        f"    <height>{h}</height>",
        "    <depth>3</depth>",
        "  </size>",
    ]
    for xmin, ymin, xmax, ymax in cajas:
        lineas += [
            "  <object>",
            "    <name>WF</name>",
            "    <bndbox>",
            f"      <xmin>{int(xmin)}</xmin>",
            f"      <ymin>{int(ymin)}</ymin>",
            f"      <xmax>{int(xmax)}</xmax>",
            f"      <ymax>{int(ymax)}</ymax>",
            "    </bndbox>",
            "  </object>",
        ]
    lineas.append("</annotation>")
    with open(ruta_xml, "w") as f:
        f.write("\n".join(lineas))


if __name__ == "__main__":
    dir_img_salida = os.path.join(DIR_SALIDA, "images")
    dir_ann_salida = os.path.join(DIR_SALIDA, "annotations")
    os.makedirs(dir_img_salida, exist_ok=True)
    os.makedirs(dir_ann_salida, exist_ok=True)

    print("--- Recortando moscas del dataset publico ---")
    recortes_publico = cargar_recortes_publico()
    print(f"{len(recortes_publico)} recortes publicos")

    print("\n--- Determinando el split propio (mismo que el resto del proyecto) ---")
    muestras_por_fuente = yt.recolectar_muestras()
    train, val, test = yt.dividir_train_val_test(muestras_por_fuente, yt.PROPORCION_VAL, yt.PROPORCION_TEST, yt.SEMILLA)
    train_propio = [m for m in train if m["grupo"] == "propio"]
    print(f"{len(train_propio)} imagenes propias de TRAIN usadas como fondo (val/test propio NUNCA se tocan)")

    print(f"\n--- Recortando moscas del PROPIO dataset (tope de {MAX_RECORTES_POR_IMAGEN_PROPIO}/imagen) ---")
    recortes_propio = cargar_recortes_propio_train(train_propio)
    print(f"{len(recortes_propio)} recortes propios")
    print(f"Mezcla: {PROB_USAR_PROPIO*100:.0f}% propio / {(1-PROB_USAR_PROPIO)*100:.0f}% publico por mosca pegada")

    total_generadas, total_cajas_nuevas = 0, 0
    total_de_propio, total_de_publico = 0, 0
    for muestra in train_propio:
        for n in range(VARIANTES_POR_IMAGEN):
            compuesta, cajas = generar_variante(
                muestra["ruta_imagen"], muestra["cajas"], recortes_publico, recortes_propio
            )
            nombre_salida = f"{muestra['nombre_salida']}_cp{n}"
            compuesta.save(os.path.join(dir_img_salida, nombre_salida + ".jpg"), "JPEG", quality=95)
            escribir_voc_xml(
                os.path.join(dir_ann_salida, nombre_salida + ".xml"),
                nombre_salida + ".jpg", compuesta.width, compuesta.height, cajas,
            )
            total_generadas += 1
            total_cajas_nuevas += len(cajas) - len(muestra["cajas"])

    print(f"\n=== Listo: {total_generadas} imagenes sinteticas generadas en {DIR_SALIDA} ===")
    print(f"Total de moscas 'pegadas' agregadas: {total_cajas_nuevas}")
    print(f"Promedio de moscas pegadas por imagen: {total_cajas_nuevas / total_generadas:.1f}")
