"""Auditoria (2026-09-14): regenera la Figura 30 (5-fold antes/despues) con AMBAS series
calculadas mediante el MISMO emparejamiento manual explicito (no modelo.val() de Ultralytics),
usando los resultados reales de recalcular_kfold_antiguo_iou_match.py (antes, Etapa 18,
modelo y etiquetas viejas) y recalcular_kfold_iou_match.py (despues, Etapa 19b, modelo y
etiquetas corregidas). Umbral iou_match=0.45 (el criterio estricto, el mismo que usa el NMS
real de la app). Mismo estilo visual que generar_graficas_tesis.py (no lo modifica)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DIR_SALIDA = rf"{RAIZ}\graficas_tesis"
os.makedirs(DIR_SALIDA, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})
VERDE, GRIS = "#1e9e6a", "#7a7d74"

with open(rf"{RAIZ}\kfold_antiguo_iou_match_resultados.json") as f:
    datos_antes = json.load(f)
with open(rf"{RAIZ}\kfold_iou_match_resultados.json") as f:
    datos_despues = json.load(f)

UMBRAL = "0.45"
f1_antes = datos_antes["resumen"][UMBRAL]["f1_por_fold"]
media_antes, std_antes = datos_antes["resumen"][UMBRAL]["media"], datos_antes["resumen"][UMBRAL]["std"]
f1_despues = datos_despues["resumen"][UMBRAL]["f1_por_fold"]
media_despues, std_despues = datos_despues["resumen"][UMBRAL]["media"], datos_despues["resumen"][UMBRAL]["std"]

folds_nombres = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]

fig, ax = plt.subplots(figsize=(9, 5.4))
x = list(range(len(folds_nombres)))
ancho = 0.34
ax.bar([i - ancho / 2 for i in x], f1_antes, width=ancho, label="Antes (Etapa 18)", color=GRIS, zorder=3)
ax.bar([i + ancho / 2 for i in x], f1_despues, width=ancho, label="Después (Etapa 19b)", color=VERDE, zorder=3)
for xi, v in zip(x, f1_antes):
    ax.text(xi - ancho / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5, color=GRIS, fontweight="bold")
for xi, v in zip(x, f1_despues):
    ax.text(xi + ancho / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5, color="#12633f", fontweight="bold")
ax.axhline(media_antes, color=GRIS, linestyle=":", linewidth=1.5, zorder=2)
ax.axhline(media_despues, color=VERDE, linestyle=":", linewidth=1.5, zorder=2)
ax.text(4.45, 0.79, f"media antes = {media_antes:.4f}±{std_antes:.4f}",
        fontsize=8.5, color=GRIS, ha="right", fontweight="bold")
ax.text(4.45, 0.745, f"media después = {media_despues:.4f}±{std_despues:.4f}",
        fontsize=8.5, color="#12633f", ha="right", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(folds_nombres)
ax.set_ylabel("F1-Score (propio, held-out del fold) — emparejamiento manual, IoU≥0,45")
ax.set_ylim(0, 0.85)
ax.legend(loc="upper left", frameon=False)
ax.set_title("Validación cruzada 5-fold — antes/después de corregir etiquetas (Etapa 18 vs 19b)\n"
             "Ambas series con emparejamiento manual explícito (mismo criterio que Candidato A)",
             fontsize=12, fontweight="bold")
ruta = os.path.join(DIR_SALIDA, "fig7_kfold_cv_corregida.png")
fig.savefig(ruta, bbox_inches="tight")
plt.close(fig)
print(f"Guardada: {ruta}")
print(f"Antes: {[round(v,4) for v in f1_antes]}, media={media_antes:.4f}±{std_antes:.4f}")
print(f"Despues: {[round(v,4) for v in f1_despues]}, media={media_despues:.4f}±{std_despues:.4f}")
