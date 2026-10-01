"""Genera la curva de entrenamiento de combinado-6_v2 (el modelo final real, etiquetas
corregidas) a partir de results.csv real de Ultralytics -- mismo estilo que fig3 (la de
combinado-6 viejo), para reemplazarla/complementarla como evidencia del modelo desplegado."""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = r"C:\Users\jchag\Documents\TESIS"
RUTA_CSV = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\results.csv"
DIR_SALIDA = rf"{RAIZ}\graficas_tesis"
os.makedirs(DIR_SALIDA, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})
AZUL, NARANJA, VERDE, GRIS = "#2a6fb0", "#d9622b", "#1e9e6a", "#7a7d74"

epocas, precision, recall, map50, map5095 = [], [], [], [], []
with open(RUTA_CSV, newline="") as f:
    for fila in csv.DictReader(f):
        epocas.append(int(fila["epoch"]))
        precision.append(float(fila["metrics/precision(B)"]))
        recall.append(float(fila["metrics/recall(B)"]))
        map50.append(float(fila["metrics/mAP50(B)"]))
        map5095.append(float(fila["metrics/mAP50-95(B)"]))

mejor_idx = max(range(len(map5095)), key=lambda i: map5095[i])
mejor_epoca = epocas[mejor_idx]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epocas, precision, label="Precisión", color=AZUL, linewidth=1.3)
ax.plot(epocas, recall, label="Recall", color=NARANJA, linewidth=1.3)
ax.plot(epocas, map50, label="mAP50", color=VERDE, linewidth=1.6)
ax.plot(epocas, map5095, label="mAP50-95", color="#c9971e", linewidth=1.3)
ax.axvline(mejor_epoca, color=GRIS, linestyle=":", linewidth=1.3)
ax.text(mejor_epoca + 1, 0.82, f"mejor época ({mejor_epoca})", fontsize=8.5, color=GRIS, va="top")
ax.axvline(epocas[-1], color="#c0392b", linestyle="--", linewidth=1.1, alpha=0.7)
ax.text(epocas[-1] - 1, 0.06, f"early stopping\n(época {epocas[-1]})", fontsize=8, color="#c0392b", ha="right")
ax.set_xlabel("Época")
ax.set_ylabel("Valor de la métrica")
ax.set_ylim(0, 0.9)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False, ncol=4)
ax.set_title("Curva de entrenamiento — combinado-6_v2 (val, etiquetas corregidas)", fontsize=13, fontweight="bold")
ruta = os.path.join(DIR_SALIDA, "fig17_curva_entrenamiento_combinado6_v2.png")
fig.savefig(ruta, bbox_inches="tight")
plt.close(fig)
print(f"Guardada: {ruta}")
print(f"Mejor epoca (max mAP50-95): {mejor_epoca} -> mAP50-95={map5095[mejor_idx]:.4f}")
print(f"Ultima epoca (early stopping): {epocas[-1]}")
