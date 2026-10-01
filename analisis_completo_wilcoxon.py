"""Completa el analisis estadistico: (1) prueba complementaria de Wilcoxon sobre falsos
positivos en las 20 imagenes SIN mosca real (donde F1 no esta definido, pero FP si es una
metrica valida), y (2) graficas: F1 pareado (24 imagenes con mosca) y FP pareado (20 imagenes
sin mosca) -- para que ninguna imagen del test publico quede fuera del analisis, solo que cada
grupo se mide con la metrica que le corresponde."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DIR_SALIDA = rf"{RAIZ}\graficas_tesis"

with open(rf"{RAIZ}\wilcoxon_publico_resultados.json") as f:
    datos = json.load(f)

filas = datos["filas"]  # nombre, n_real, tp_a, fp_a, fn_a, f1_a, tp_b, fp_b, fn_b, f1_b

con_mosca = [f for f in filas if f[1] > 0]
sin_mosca = [f for f in filas if f[1] == 0]
print(f"Con mosca real: {len(con_mosca)}, sin mosca real: {len(sin_mosca)}")

# ==========================================================================
# Prueba complementaria: falsos positivos por imagen SIN mosca real
# ==========================================================================
fp_a_vacias = [f[3] for f in sin_mosca]
fp_b_vacias = [f[7] for f in sin_mosca]
n_favor_b_fp = sum(1 for a, b in zip(fp_a_vacias, fp_b_vacias) if b < a)
n_favor_a_fp = sum(1 for a, b in zip(fp_a_vacias, fp_b_vacias) if b > a)
n_empate_fp = sum(1 for a, b in zip(fp_a_vacias, fp_b_vacias) if b == a)

print(f"\nFalsos positivos en imagenes sin mosca real (n={len(sin_mosca)}):")
print(f"  Total FP Candidato A: {sum(fp_a_vacias)} (media {sum(fp_a_vacias)/len(fp_a_vacias):.2f}/img)")
print(f"  Total FP Candidato B: {sum(fp_b_vacias)} (media {sum(fp_b_vacias)/len(fp_b_vacias):.2f}/img)")
print(f"  Imagenes donde B tiene MENOS falsos positivos que A: {n_favor_b_fp}")
print(f"  Imagenes donde A tiene menos: {n_favor_a_fp}, empate: {n_empate_fp}")

stat_fp, p_fp = wilcoxon(fp_a_vacias, fp_b_vacias, alternative="greater")  # H1: FP_A > FP_B
print(f"  Wilcoxon (H1: FP_A > FP_B), una cola: estadistico={stat_fp:.2f}, p={p_fp:.5f}")

# ==========================================================================
# Figura 19a: F1 pareado por imagen (las 24 imagenes CON mosca real)
# ==========================================================================
f1s_a = [f[5] for f in con_mosca]
f1s_b = [f[9] for f in con_mosca]
orden = sorted(range(len(con_mosca)), key=lambda i: f1s_a[i])
f1s_a_o = [f1s_a[i] for i in orden]
f1s_b_o = [f1s_b[i] for i in orden]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})
AZUL, NARANJA, GRIS = "#2a6fb0", "#d9622b", "#7a7d74"

fig, ax = plt.subplots(figsize=(9, 6))
x = range(len(con_mosca))
for i in x:
    color = "#1e9e6a" if f1s_b_o[i] > f1s_a_o[i] else "#c0392b"
    ax.plot([i, i], [f1s_a_o[i], f1s_b_o[i]], color=color, linewidth=1.2, alpha=0.6, zorder=1)
ax.scatter(x, f1s_a_o, color=AZUL, s=45, zorder=3, label="Candidato A (clásico)")
ax.scatter(x, f1s_b_o, color=NARANJA, s=45, zorder=3, label="Candidato B (YOLOv8n)")
ax.set_xlabel("Imágenes de test público (n=24, ordenadas por F1 de A)")
ax.set_ylabel("F1-Score por imagen")
ax.set_ylim(-0.02, 1.05)
ax.set_xticks([])
ax.legend(loc="lower right", frameon=False)
ax.set_title("F1 pareado por imagen — Candidato A vs. B\n(Wilcoxon, dos colas: p=0.00004)",
              fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{DIR_SALIDA}\\fig19a_wilcoxon_f1_pareado.png", bbox_inches="tight")
plt.close(fig)
print(f"\nGuardada: {DIR_SALIDA}\\fig19a_wilcoxon_f1_pareado.png")

# ==========================================================================
# Figura 19b: Falsos positivos pareados en imagenes SIN mosca real (n=20)
# ==========================================================================
orden_fp = sorted(range(len(sin_mosca)), key=lambda i: fp_a_vacias[i])
fp_a_o = [fp_a_vacias[i] for i in orden_fp]
fp_b_o = [fp_b_vacias[i] for i in orden_fp]

fig, ax = plt.subplots(figsize=(9, 6))
x = range(len(sin_mosca))
ancho = 0.38
ax.bar([i - ancho / 2 for i in x], fp_a_o, width=ancho, color=AZUL, label="Candidato A (clásico)", zorder=3)
ax.bar([i + ancho / 2 for i in x], fp_b_o, width=ancho, color=NARANJA, label="Candidato B (YOLOv8n)", zorder=3)
ax.set_xlabel("Imágenes de test público sin mosca real (n=20, ordenadas por FP de A)")
ax.set_ylabel("Falsos positivos por imagen")
ax.set_xticks([])
ax.legend(loc="upper right", frameon=False)
ax.set_title(f"Falsos positivos en imágenes sin mosca real — Candidato A vs. B\n"
             f"(total: A={sum(fp_a_vacias)}, B={sum(fp_b_vacias)}; Wilcoxon una cola: p={p_fp:.5f})",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{DIR_SALIDA}\\fig19b_wilcoxon_fp_vacias.png", bbox_inches="tight")
plt.close(fig)
print(f"Guardada: {DIR_SALIDA}\\fig19b_wilcoxon_fp_vacias.png")

datos["fp_sin_mosca"] = {
    "n": len(sin_mosca), "total_fp_a": sum(fp_a_vacias), "total_fp_b": sum(fp_b_vacias),
    "n_favor_b": n_favor_b_fp, "n_favor_a": n_favor_a_fp, "n_empate": n_empate_fp,
    "wilcoxon_stat": float(stat_fp), "wilcoxon_p": float(p_fp),
}
with open(rf"{RAIZ}\wilcoxon_publico_resultados.json", "w") as f:
    json.dump(datos, f, indent=2)
print("\nActualizado: wilcoxon_publico_resultados.json (agregado bloque fp_sin_mosca)")
