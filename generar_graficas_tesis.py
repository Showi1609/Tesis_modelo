"""
Genera las graficas estadisticas para el documento de tesis (Word), como PNG de alta resolucion,
a partir de los datos ya calculados en toda la sesion (bitacora). No corre inferencia ni
entrenamiento -- solo grafica numeros que ya tenemos.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DIR_SALIDA = rf"{RAIZ}\graficas_tesis"
os.makedirs(DIR_SALIDA, exist_ok=True)

# estilo general, limpio y apto para imprimir
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

AZUL = "#2a6fb0"
NARANJA = "#d9622b"
VERDE = "#1e9e6a"
GRIS = "#7a7d74"
ROJO = "#c0392b"


def guardar(fig, nombre):
    ruta = os.path.join(DIR_SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardada: {ruta}")


# ==========================================================================
# Figura 1: Evolucion historica de F1 propio en el pipeline
# ==========================================================================
etapas_1 = ["Sin sobre-\nmuestreo\n(Etapa 3b)", "Con sobre-\nmuestreo\n(Etapa 4)",
            "Fine-tuning\npropio\n(Etapa 5)", "Copy-paste\noriginal\n(Etapa 6)",
            "Copy-paste +\netiquetas corr.\n(Etapa 15)"]
valores_1 = [0.4624, 0.5992, 0.6250, 0.6325, 0.7578]

fig, ax = plt.subplots(figsize=(8, 4.8))
x = range(len(etapas_1))
barras = ax.bar(x, valores_1, color=AZUL, width=0.55, zorder=3)
ax.axhline(0.6455, color=GRIS, linestyle=":", linewidth=1.5, zorder=2)
ax.text(0.35, 0.6455 + 0.015, "Clásico (Candidato A) — 0.6455 (dataset público completo, referencia)",
        fontsize=8.5, color=GRIS, ha="left")
for xi, v in zip(x, valores_1):
    ax.text(xi, v + 0.015, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(etapas_1, fontsize=9)
ax.set_ylabel("F1-Score (propio)")
ax.set_ylim(0, 0.85)
ax.set_title("Evolución de F1 propio a lo largo del pipeline", fontsize=13, fontweight="bold")
guardar(fig, "fig1_evolucion_f1_propio.png")

# ==========================================================================
# Figura 2: Domain gap antes/despues
# ==========================================================================
etapas_2 = ["Con sobremuestreo\n(Etapa 4/10)", "Copy-paste\n(Etapa 6/10)", "Copy-paste +\netiquetas corregidas\n(Etapa 15)"]
publico_2 = [0.8043, 0.8149, 0.8149]
propio_2 = [0.5992, 0.6325, 0.7578]
gap_2 = [p - r for p, r in zip(publico_2, propio_2)]

fig, ax = plt.subplots(figsize=(8, 4.8))
x = range(len(etapas_2))
ancho = 0.32
ax.bar([i - ancho / 2 for i in x], publico_2, width=ancho, label="F1 público", color=AZUL, zorder=3)
ax.bar([i + ancho / 2 for i in x], propio_2, width=ancho, label="F1 propio", color=NARANJA, zorder=3)
for xi, (p, r, g) in enumerate(zip(publico_2, propio_2, gap_2)):
    if g > 0.06:  # solo dibuja la flecha si hay espacio suficiente para que se vea bien
        ax.annotate("", xy=(xi, p - 0.015), xytext=(xi, r + 0.015),
                    arrowprops=dict(arrowstyle="<->", color=ROJO, lw=1.6))
    ax.text(xi + 0.38, (p + r) / 2, f"gap\n{g*100:.1f} pts", fontsize=9, color=ROJO, va="center")
ax.set_xticks(list(x))
ax.set_xticklabels(etapas_2, fontsize=9.5)
ax.set_ylabel("F1-Score")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left", frameon=False)
ax.set_title("Domain gap (público vs. propio) antes y después de las correcciones", fontsize=13, fontweight="bold")
guardar(fig, "fig2_domain_gap.png")

# ==========================================================================
# Figura 3: Curva de entrenamiento de combinado-6 (89 epocas)
# ==========================================================================
datos_d = [
    (1,0.03549,0.39405,0.10154,0.03115),(2,0.53796,0.53556,0.44866,0.15550),(3,0.63655,0.56967,0.54124,0.18785),
    (4,0.48229,0.53411,0.34590,0.11024),(5,0.61204,0.56749,0.49669,0.16233),(6,0.48415,0.52975,0.38876,0.13421),
    (7,0.58411,0.55515,0.50662,0.16610),(8,0.60845,0.56676,0.53064,0.18678),(9,0.60037,0.54572,0.52669,0.17922),
    (10,0.64057,0.57547,0.55510,0.19623),(11,0.70983,0.59652,0.60109,0.21563),(12,0.61218,0.54717,0.52383,0.19538),
    (13,0.60361,0.64659,0.53715,0.18885),(14,0.65169,0.63570,0.57581,0.20205),(15,0.50346,0.54935,0.43773,0.15103),
    (16,0.64255,0.63135,0.58629,0.20979),(17,0.61119,0.61601,0.55997,0.20638),(18,0.55306,0.60524,0.52290,0.19081),
    (19,0.67258,0.65143,0.60968,0.21140),(20,0.56721,0.56168,0.50324,0.18769),(21,0.59530,0.57257,0.51380,0.17878),
    (22,0.61718,0.61656,0.59791,0.21947),(23,0.52067,0.56241,0.46920,0.15222),(24,0.60391,0.57757,0.56063,0.21253),
    (25,0.67421,0.59470,0.59241,0.21826),(26,0.63432,0.55766,0.52875,0.17895),(27,0.50890,0.52395,0.44753,0.16144),
    (28,0.65627,0.56459,0.54595,0.18176),(29,0.63509,0.62046,0.55887,0.19945),(30,0.53648,0.60232,0.44990,0.14123),
    (31,0.57997,0.53809,0.47279,0.15110),(32,0.68884,0.61176,0.60078,0.21805),(33,0.57216,0.54862,0.51421,0.19364),
    (34,0.58756,0.56459,0.53339,0.20044),(35,0.71726,0.63933,0.62819,0.23271),(36,0.65281,0.62084,0.60546,0.23491),
    (37,0.64723,0.62842,0.55937,0.18332),(38,0.69788,0.55370,0.57315,0.19562),(39,0.57774,0.54862,0.47820,0.15919),
    (40,0.58509,0.56591,0.51136,0.17860),(41,0.62487,0.61974,0.57122,0.20967),(42,0.68826,0.61030,0.62205,0.24413),
    (43,0.65832,0.58999,0.60268,0.23264),(44,0.58700,0.51016,0.48823,0.17102),(45,0.53160,0.54209,0.49282,0.18331),
    (46,0.66473,0.62700,0.59752,0.21536),(47,0.60796,0.55705,0.52030,0.18372),(48,0.54622,0.56459,0.46900,0.14380),
    (49,0.64654,0.62917,0.55301,0.18851),(50,0.57168,0.59869,0.52441,0.18120),(51,0.61960,0.58628,0.55948,0.20159),
    (52,0.58546,0.57293,0.51475,0.17384),(53,0.67408,0.67344,0.63087,0.23308),(54,0.66419,0.58636,0.59814,0.22227),
    (55,0.69934,0.64731,0.64881,0.24568),(56,0.63373,0.64789,0.60307,0.23193),(57,0.63088,0.59434,0.56666,0.21409),
    (58,0.53342,0.61311,0.52942,0.18750),(59,0.60916,0.60160,0.55854,0.20139),(60,0.73604,0.64350,0.66624,0.25678),
    (61,0.69755,0.59583,0.60719,0.23086),(62,0.57682,0.58999,0.53484,0.19484),(63,0.70036,0.65820,0.63713,0.23061),
    (64,0.65942,0.59869,0.57062,0.19738),(65,0.67021,0.62046,0.60970,0.22829),(66,0.59081,0.61924,0.57383,0.21805),
    (67,0.69726,0.64296,0.62639,0.23769),(68,0.57476,0.59047,0.52447,0.19261),(69,0.73308,0.65312,0.66875,0.26161),
    (70,0.61025,0.63403,0.57259,0.21178),(71,0.65017,0.60962,0.58674,0.22933),(72,0.71810,0.63498,0.64677,0.24077),
    (73,0.69375,0.64114,0.63921,0.25131),(74,0.65806,0.61393,0.60267,0.22839),(75,0.66001,0.55926,0.56631,0.20979),
    (76,0.65185,0.62772,0.61125,0.23752),(77,0.72374,0.64224,0.64796,0.24375),(78,0.62560,0.55588,0.53794,0.20007),
    (79,0.71870,0.61176,0.63321,0.24542),(80,0.65924,0.58781,0.59416,0.23023),(81,0.70690,0.64586,0.63742,0.24016),
    (82,0.69415,0.61321,0.62079,0.24159),(83,0.68865,0.64847,0.64324,0.24887),(84,0.62719,0.57257,0.54062,0.19605),
    (85,0.65935,0.58292,0.56599,0.22258),(86,0.74647,0.67054,0.67778,0.26042),(87,0.70321,0.63570,0.62485,0.23955),
    (88,0.63609,0.59434,0.58172,0.22356),(89,0.69338,0.59579,0.61449,0.23756),
]
epocas = [d[0] for d in datos_d]
precision = [d[1] for d in datos_d]
recall = [d[2] for d in datos_d]
map50 = [d[3] for d in datos_d]
map5095 = [d[4] for d in datos_d]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epocas, precision, label="Precisión", color=AZUL, linewidth=1.3)
ax.plot(epocas, recall, label="Recall", color=NARANJA, linewidth=1.3)
ax.plot(epocas, map50, label="mAP50", color=VERDE, linewidth=1.6)
ax.plot(epocas, map5095, label="mAP50-95", color="#c9971e", linewidth=1.3)
mejor_epoca = 69
ax.axvline(mejor_epoca, color=GRIS, linestyle=":", linewidth=1.3)
ax.text(mejor_epoca + 1, 0.79, f"mejor época ({mejor_epoca})", fontsize=8.5, color=GRIS, va="top")
ax.set_xlabel("Época")
ax.set_ylabel("Valor de la métrica")
ax.set_ylim(0, 0.85)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False, ncol=4)
ax.set_title("Curva de entrenamiento — combinado-6 (val, 51 img)", fontsize=13, fontweight="bold")
guardar(fig, "fig3_curva_entrenamiento_combinado6.png")

# ==========================================================================
# Figura 4: Comparacion de variantes de copy-paste (F1 propio, etiquetas corregidas)
# ==========================================================================
modelos_4 = ["combinado-6\n(candidato final)", "combinado-7\n(más denso)", "combinado-8\n(recortes propios)",
             "combinado-9\n(mezcla 50/50)", "combinado-10\n(+ negativos)", "alpha=0.50\n(weight soup)"]
valores_4 = [0.7578, 0.6121, 0.6941, 0.7043, 0.6918, 0.7570]
colores_4 = [VERDE, ROJO, NARANJA, NARANJA, ROJO, AZUL]

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(modelos_4))
barras = ax.bar(x, valores_4, color=colores_4, width=0.6, zorder=3)
for xi, v in zip(x, valores_4):
    ax.text(xi, v + 0.012, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(modelos_4, fontsize=9.5)
ax.set_ylabel("F1-Score (propio, etiquetas corregidas)")
ax.set_ylim(0, 0.85)
ax.set_title("Comparación de variantes exploradas — F1 propio", fontsize=13, fontweight="bold")
guardar(fig, "fig4_comparacion_variantes.png")

# ==========================================================================
# Figura 5: Cuantizacion TFLite (Etapa 8)
# ==========================================================================
variantes_5 = ["FP32", "FP16", "Dynamic-range\nINT8 (elegida)", "Static INT8"]
tamano_5 = [12.9, 6.5, 4.0, 3.4]
f1_5 = [0.723, 0.724, 0.727, 0.250]
colores_5 = [GRIS, GRIS, VERDE, ROJO]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
ax1.bar(variantes_5, tamano_5, color=colores_5, zorder=3)
for i, v in enumerate(tamano_5):
    ax1.text(i, v + 0.2, f"{v} MB", ha="center", fontsize=9.5, fontweight="bold")
ax1.set_ylabel("Tamaño (MB)")
ax1.set_title("Tamaño del modelo", fontsize=11.5)
ax1.tick_params(axis="x", labelsize=8.5)

ax2.bar(variantes_5, f1_5, color=colores_5, zorder=3)
for i, v in enumerate(f1_5):
    ax2.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
ax2.set_ylabel("F1-Score (test agregado)")
ax2.set_ylim(0, 0.85)
ax2.set_title("F1 tras cuantizar", fontsize=11.5)
ax2.tick_params(axis="x", labelsize=8.5)

fig.suptitle("Cuantización TFLite — 4 variantes (Etapa 8)", fontsize=13, fontweight="bold", y=1.03)
guardar(fig, "fig5_cuantizacion_tflite.png")

# ==========================================================================
# Figura 6: Bootstrap -- incertidumbre de F1 propio (Etapa 14, con etiquetas corregidas)
# ==========================================================================
import json

with open(rf"{RAIZ}\bootstrap_f1_propio_raw.json") as f:
    boot = json.load(f)

f1s = boot["f1s"]
media = boot["media"]
ci_low, ci_high = boot["ci95"]

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.hist(f1s, bins=40, color=AZUL, alpha=0.75, zorder=3, edgecolor="white", linewidth=0.4)
ax.axvline(media, color=ROJO, linewidth=2, label=f"Media bootstrap = {media:.4f}")
ax.axvspan(ci_low, ci_high, color=NARANJA, alpha=0.15, label=f"IC 95% = [{ci_low:.3f}, {ci_high:.3f}]")
ax.axvline(ci_low, color=NARANJA, linestyle="--", linewidth=1.3)
ax.axvline(ci_high, color=NARANJA, linestyle="--", linewidth=1.3)
ax.set_xlabel("F1-Score (propio) por remuestreo")
ax.set_ylabel("Frecuencia (de 1000 remuestreos)")
ax.legend(loc="upper left", frameon=False, fontsize=9.5)
ax.set_title("Bootstrap de F1 propio — combinado-6 (etiquetas corregidas, Etapa 15)", fontsize=12.5, fontweight="bold")
guardar(fig, "fig6_bootstrap_f1_propio.png")

# ==========================================================================
# Figura 14: Barrido de alpha -- weight averaging (Etapa 9)
# ==========================================================================
alphas = [0.00, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00]
f1_publico_a = [0.8113, 0.8029, 0.7888, 0.7387, 0.6824, 0.6506, 0.6227]
f1_propio_a = [0.6188, 0.6179, 0.6215, 0.6410, 0.6375, 0.6298, 0.6254]
gap_a = [p - r for p, r in zip(f1_publico_a, f1_propio_a)]

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(alphas, f1_publico_a, "o-", label="F1 público", color=AZUL, linewidth=2, markersize=6)
ax.plot(alphas, f1_propio_a, "o-", label="F1 propio", color=NARANJA, linewidth=2, markersize=6)
ax.plot(alphas, gap_a, "o--", label="Domain gap", color=ROJO, linewidth=1.6, markersize=5, alpha=0.8)
ax.axvline(0.0, color=GRIS, linestyle=":", linewidth=1)
ax.axvline(1.0, color=GRIS, linestyle=":", linewidth=1)
ax.text(0.02, 0.05, "combinado-6\npuro", fontsize=8, color=GRIS)
ax.text(0.90, 0.05, "finetune_v2\npuro", fontsize=8, color=GRIS, ha="right")
ax.set_xlabel("alpha (peso de finetune_propio_v2 en la mezcla)")
ax.set_ylabel("F1-Score / Gap")
ax.set_ylim(0, 0.9)
ax.legend(loc="upper right", frameon=False)
ax.set_title("Weight averaging — barrido de alpha (Etapa 9)", fontsize=13, fontweight="bold")
guardar(fig, "fig14_barrido_alpha.png")

# ==========================================================================
# Figura 15: Barrido SAHI vs. baseline (Etapa 16)
# ==========================================================================
conf_s = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
p_s = [0.5768, 0.6507, 0.7098, 0.7743, 0.8043, 0.8793, 0.9286, 0.9583, 0.9565, 1.0000]
r_s = [0.8613, 0.8347, 0.7893, 0.7227, 0.5920, 0.4080, 0.2427, 0.1227, 0.0587, 0.0160]
f1_s = [0.6909, 0.7313, 0.7475, 0.7476, 0.6820, 0.5574, 0.3848, 0.2175, 0.1106, 0.0315]

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(conf_s, p_s, "o-", label="Precisión (SAHI)", color=AZUL, linewidth=1.6, markersize=5)
ax.plot(conf_s, r_s, "o-", label="Recall (SAHI)", color=NARANJA, linewidth=1.6, markersize=5)
ax.plot(conf_s, f1_s, "o-", label="F1 (SAHI)", color=VERDE, linewidth=2.2, markersize=6)
ax.axhline(0.7578, color=ROJO, linestyle="--", linewidth=1.6, label="F1 baseline (sin SAHI, conf=0.25) = 0.7578")
ax.axvline(0.40, color=GRIS, linestyle=":", linewidth=1.2)
ax.text(0.405, 0.05, "mejor punto\nSAHI (0.40)", fontsize=8, color=GRIS)
ax.set_xlabel("Umbral de confianza")
ax.set_ylabel("Valor de la métrica")
ax.set_ylim(0, 1.02)
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_title("SAHI — barrido de confianza vs. baseline (Etapa 16)", fontsize=13, fontweight="bold")
guardar(fig, "fig15_barrido_sahi.png")

print("\n=== Listo: 8 graficas custom generadas en", DIR_SALIDA, "===")
print("(+ 6 oficiales de Ultralytics copiadas: fig8-fig13. La fig7 del 5-fold, pendiente)")

# ==========================================================================
# Figura 7: 5-fold CV -- ANTES (Etapa 18) vs DESPUES (Etapa 19b) de corregir etiquetas
# ==========================================================================
folds_nombres = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
f1_antes = [0.5606, 0.5494, 0.3927, 0.4598, 0.5175]
f1_despues = [0.5465, 0.6676, 0.4939, 0.6896, 0.6652]
media_antes, std_antes = 0.4960, 0.0624
media_despues, std_despues = 0.6126, 0.0777

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
ax.set_ylabel("F1-Score (propio, held-out del fold)")
ax.set_ylim(0, 0.85)
ax.legend(loc="upper left", frameon=False)
ax.set_title("Validación cruzada 5-fold — antes/después de corregir etiquetas (Etapa 18 vs 19b)",
             fontsize=12.5, fontweight="bold")
guardar(fig, "fig7_kfold_cv.png")
print("Guardada fig7 -- ya estan las 9 graficas custom completas (15 en total con las 6 oficiales)")

# ==========================================================================
# Figura 16: Candidato A vs B -- ambos dominios, ambos umbrales de matching (Etapa 20)
# ==========================================================================
grupos = ["A — propio", "A — público", "B — propio", "B — público"]
f1_permisivo = [0.6535, 0.6453, 0.6990, 0.8499]
f1_estricto = [0.1580, 0.0263, 0.6737, 0.8126]
colores_grupo = [ROJO, ROJO, VERDE, VERDE]

fig, ax = plt.subplots(figsize=(10, 5.8))
x = list(range(len(grupos)))
ancho = 0.32
barras1 = ax.bar([i - ancho/2 for i in x], f1_permisivo, width=ancho, color=colores_grupo,
                  alpha=0.55, zorder=3, label="iou=0.1 (permisivo)")
barras2 = ax.bar([i + ancho/2 for i in x], f1_estricto, width=ancho, color=colores_grupo,
                  zorder=3, label="iou=0.45 (estricto)")
for xi, v in zip(x, f1_permisivo):
    ax.text(xi - ancho/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
for xi, v in zip(x, f1_estricto):
    ax.text(xi + ancho/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

for xi, (vp, ve) in enumerate(zip(f1_permisivo, f1_estricto)):
    caida = (vp - ve) * 100
    ax.text(xi, max(vp, ve) + 0.06, f"-{caida:.0f}pts", ha="center", fontsize=9, color=GRIS, fontweight="bold")

# leyenda manual de colores por candidato (aparte de la de los patrones de umbral)
from matplotlib.patches import Patch
legend_umbral = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
ax.add_artist(legend_umbral)
legend_candidato = [Patch(facecolor=ROJO, label="Candidato A (clásico)"),
                     Patch(facecolor=VERDE, label="Candidato B (YOLO)")]
ax.legend(handles=legend_candidato, loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False, fontsize=9.5)

ax.set_xticks(x)
ax.set_xticklabels(grupos, fontsize=10.5)
ax.set_ylabel("F1-Score")
ax.set_ylim(0, 1.0)
fig.suptitle("Candidato A vs. B — ambos dominios, ambos umbrales de matching", fontsize=13, fontweight="bold", x=0.53)
ax.set_title("YOLO es estable ante el criterio de evaluación; el clásico colapsa con iou estricto (Etapa 20)",
             fontsize=9.5, color=GRIS, pad=10)
guardar(fig, "fig16_candidato_a_vs_b.png")
print("Guardada fig16 -- comparacion final Candidato A vs B, ambos dominios y umbrales (10 graficas custom, 16 en total)")
