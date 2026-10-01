"""Genera una figura de 'antes/despues' del copy-paste augmentation: la imagen propia
original con sus cajas reales, junto a la variante sintetica con las moscas pegadas
resaltadas en otro color -- para mostrar visualmente la tecnica en el documento."""

import os
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

RAIZ = r"C:\Users\jchag\Documents\TESIS"
NOMBRE_BASE = "2007_jpg.rf.aK0DTVLoOKSvEvzkPMGJ"
RUTA_ORIG_IMG = rf"{RAIZ}\Propio\Tesis.voc\imagess\{NOMBRE_BASE}.jpg"
RUTA_ORIG_XML = rf"{RAIZ}\Propio\Tesis.voc\annotations\{NOMBRE_BASE}.xml"
RUTA_CP_IMG = rf"{RAIZ}\Propio_copypaste\images\propio_{NOMBRE_BASE}_cp0.jpg"
RUTA_CP_XML = rf"{RAIZ}\Propio_copypaste\annotations\propio_{NOMBRE_BASE}_cp0.xml"
DIR_SALIDA = rf"{RAIZ}\graficas_tesis"


def leer_cajas(ruta_xml):
    cajas = []
    root = ET.parse(ruta_xml).getroot()
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        cajas.append((
            float(b.find("xmin").text), float(b.find("ymin").text),
            float(b.find("xmax").text), float(b.find("ymax").text),
        ))
    return cajas


cajas_orig = leer_cajas(RUTA_ORIG_XML)
cajas_cp = leer_cajas(RUTA_CP_XML)
n_originales = len(cajas_orig)
cajas_sinteticas = cajas_cp[n_originales:]  # el script las agrega al final de la lista

img_orig = Image.open(RUTA_ORIG_IMG).convert("RGB")
img_cp = Image.open(RUTA_CP_IMG).convert("RGB")

# --- Recorte cerrado sobre la zona con cajas (trampa), con margen generoso ---
todas_cajas = cajas_orig + cajas_sinteticas
xs0 = [c[0] for c in todas_cajas]; ys0 = [c[1] for c in todas_cajas]
xs1 = [c[2] for c in todas_cajas]; ys1 = [c[3] for c in todas_cajas]
x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
w, h = x1 - x0, y1 - y0
mx, my = w * 0.25, h * 0.25  # margen del 25% alrededor del area con moscas
cx0 = max(0, x0 - mx); cy0 = max(0, y0 - my)
cx1 = min(img_orig.width, x1 + mx); cy1 = min(img_orig.height, y1 + my)
recorte = (int(cx0), int(cy0), int(cx1), int(cy1))

img_orig_c = img_orig.crop(recorte)
img_cp_c = img_cp.crop(recorte)
ox, oy = recorte[0], recorte[1]  # offset a restar a las coordenadas de las cajas


def dibujar_cajas(ax, cajas, color):
    for (bx0, by0, bx1, by1) in cajas:
        rx0, ry0 = bx0 - ox, by0 - oy
        rw, rh = bx1 - bx0, by1 - by0
        # caja fina real + marco grueso semi-transparente alrededor para que resalte
        ax.add_patch(patches.Rectangle((rx0 - 6, ry0 - 6), rw + 12, rh + 12, linewidth=2.4,
                                        edgecolor=color, facecolor="none", alpha=0.95))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.5))

ax1.imshow(img_orig_c)
dibujar_cajas(ax1, cajas_orig, "#1e9e3f")
ax1.set_title(f"Original — {n_originales} mosca(s) real(es)", fontsize=13, fontweight="bold")
ax1.axis("off")

ax2.imshow(img_cp_c)
dibujar_cajas(ax2, cajas_orig, "#1e9e3f")
dibujar_cajas(ax2, cajas_sinteticas, "#e0651f")
ax2.set_title(f"Variante sintética — +{len(cajas_sinteticas)} mosca(s) pegada(s)", fontsize=13, fontweight="bold")
ax2.axis("off")

verde = patches.Patch(edgecolor="#1e9e3f", facecolor="none", linewidth=2.4, label="Mosca real (original)")
naranja = patches.Patch(edgecolor="#e0651f", facecolor="none", linewidth=2.4, label="Mosca sintética (pegada de público)")
fig.legend(handles=[verde, naranja], loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=12)

fig.suptitle("Copy-paste augmentation: fondo propio + recortes reales del dataset público",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0.04, 1, 0.95])

ruta = os.path.join(DIR_SALIDA, "fig18_copypaste_antes_despues.png")
fig.savefig(ruta, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Guardada: {ruta}")
print(f"Originales: {n_originales}, sinteticas agregadas: {len(cajas_sinteticas)}")
print(f"Recorte: {recorte} sobre imagen {img_orig.size}")
