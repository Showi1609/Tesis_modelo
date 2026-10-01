# Pipeline de Entrenamiento, Evaluación y Despliegue — BioCount MIP (Modelo v2)

Este repositorio contiene el código fuente, conjuntos de datos y resultados numéricos del modelo de visión por computador para la detección y conteo automático de mosca blanca (*Trialeurodes vaporariorum* / *Bemisia tabaci*) en trampas cromáticas amarillas.

---

## 1. Especificaciones del Modelo Final (`combinado-6_v2`)

- **Arquitectura:** YOLOv8n (nano) adaptado para detección de objetos pequeños.
- **Modelo desplegado:** `models/whitefly_yolov8n_combinado6_v2_int8.tflite` (cuantizado a INT8).
- **Suma de comprobación SHA-256:** `a4be7a54d71861a2a48e83630c0516dfa96c2b29927b7e42280587e20e388fa4`
- **Parámetros de Inferencia en Aplicación Móvil:**
  - Tamaño de entrada: 1280 px (con letterbox gris RGB 114, 114, 114).
  - Umbral de confianza (`conf`): 0.25 (fijo).
  - Umbral de NMS IoU (`iou`): 0.45 (fijo).
  - Modo por defecto: Mosaico 2×2 (20% de traslape, fusión por contención 0.65).

---

## 2. Definición de Candidatos Comparados

- **Candidato A (Pipeline Clásico con OpenCV):** Pipeline tradicional basado en visión por computador clásica (espacios de color, umbralización adaptable, operaciones morfológicas y filtrado de contornos con OpenCV).
- **Candidato B (YOLOv8n `combinado-6_v2` INT8 TFLite):** Modelo de aprendizaje profundo basado en YOLOv8n entrenado con el conjunto de datos combinado (`md121` + dataset propio con aumento Copy-Paste (`copy_paste_augmentation.py`)) y cuantizado a INT8.

---

## 3. Orden de Reproducción de Experimentos

1. **Preparación de Datasets:**
   - Datasets fuente en `md121/` y `Propio/Tesis.voc/`.
2. **Aumento de Datos:**
   - Script: `copy_paste_augmentation.py` (Genera el conjunto propio aumentado con Copy-Paste).
3. **Entrenamiento:**
   - Scripts: `yolo_entrenamiento.py` y `yolo_entrenamiento_combinado6_v2.py`.
4. **Validación Cruzada 5-Fold (5-Fold CV):**
   - Scripts: `kfold_cv_v2.py` y `recalcular_kfold_iou_match.py`.
   - Resultados exportados en: `kfold_iou_match_resultados.json` y `kfold_antiguo_iou_match_resultados.json`.
5. **Evaluación de Candidatos:**
   - Script Candidato A (Público 44 imágenes): `candidato_a_publico_fix_exif_iou01.py` y `reevaluar_candidato_a_publico44.py`.
     - Resultados exportados en: `candidato_a_publico44.json`.
   - Script Candidato A (Dataset propio): `reevaluar_candidato_a_propio.py`.
     - Resultados exportados en: `candidato_a_propio_iou045.json`.
   - Scripts Candidato B (YOLOv8n v2): `evaluar_combinado6_v2_iou_match.py`, `exportar_tflite_v2.py`, `evaluar_tflite_v2.py`.
6. **Pruebas Estadísticas:**
   - Script: `test_wilcoxon_publico.py` y `analisis_completo_wilcoxon.py`.
   - Resultados exportados en: `wilcoxon_publico_resultados.json` y `wilcoxon_propio_definitivo.json`.
   - Datos del bootstrap: `bootstrap_f1_propio_raw.json` (entrada de `generar_graficas_tesis.py`).
7. **Generación de Gráficas de la Tesis:**
   - Scripts: `generar_graficas_tesis.py`, `generar_fig_curva_v2.py`, `generar_fig_copypaste.py`, `generar_fig_kfold_corregida.py`.
   - Salidas visuales en: `graficas_tesis/`.

---

## 4. Resultados Clave Citados (Origen de Datos)

Todos los valores citados provienen de archivos JSON de resultados versionados en este repositorio:

| Experimento / Conjunto | Archivo JSON de Origen | Métrica Destacada |
| :--- | :--- | :--- |
| **Candidato A — Público (44 imágenes)** | `candidato_a_publico44.json` | **IoU 0.10:** F1 = 0.5361 (TP=338, FP=410, FN=175)<br>**IoU 0.45:** F1 = 0.0539 (TP=34, FP=714, FN=479) |
| **Prueba de Wilcoxon (Público - 24 pares)** | `wilcoxon_publico_resultados.json` | F1 medio A = 0.4964 vs F1 medio B = 0.8199<br>Gana B: 23, Gana A: 1 ($p = 1.82 \times 10^{-5}$) |
| **Prueba de Wilcoxon (Propio - 60 imágenes), IoU 0.10** | `wilcoxon_propio_definitivo.json` | F1 medio A = 0.588 vs F1 medio B = 0.6444<br>Gana B: 34, Gana A: 24, Empates: 2 ($p = 0.0437$) |
| **Prueba de Wilcoxon (Propio - 60 imágenes)** | `wilcoxon_propio_definitivo.json` | **IoU 0.45:** F1 medio A = 0.0957 vs F1 medio B = 0.6009<br>Gana B: 58, Gana A: 1 ($p = 1.88 \times 10^{-10}$) |
| **Validación Cruzada 5-Fold** | `kfold_iou_match_resultados.json` | Métricas agregadas por pliegue de la validación cruzada. |

*Nota: Los scripts conservan las rutas absolutas del entorno donde se ejecutaron; para reproducirlos, ajuste las rutas `BASE` al inicio de cada archivo.*
