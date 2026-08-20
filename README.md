# BioCount MIPE — Modelo (YOLOv8n, detección de mosca blanca)

Pipeline de entrenamiento y evaluación del Candidato B (YOLOv8n) para el conteo automático de
mosca blanca (*whitefly*, WF) en trampas amarillas pegajosas, parte del proyecto de tesis
BioCount MIPE. Complementa al repo de la app Android que consume el modelo exportado.

Bitácora completa del proceso (10 etapas, métricas por época, decisiones y descartes):
**[link al artifact — actualizar con la URL pública]**

## Resultado final

Modelo recomendado: `models/combinado-6_best.pt` (F1 agregado test = 0.7445, público = 0.8149,
propio = 0.6325, medido con el NMS iou=0.45 real de la app — ver Etapa 10 de la bitácora).
Exportado a `models/combinado-6_best_int8.tflite` (cuantización dynamic-range, 4MB) para la app.

## Estructura

```
├── yolo_entrenamiento.py              # pipeline principal: dataset + entrenamiento (combinado-6)
├── yolo_entrenamiento_combinado7.py   # variante con copy-paste mas denso (Etapa 11, en evaluación)
├── copy_paste_augmentation.py         # genera las 126 imágenes sintéticas (Etapa 6)
├── copy_paste_augmentation_v2.py      # variante con más moscas por trampa sintética (Etapa 11)
├── finetune_propio.py                 # fine-tuning sobre propio (Etapas 5 y 7)
├── weight_soup_propio.py              # interpolación de pesos combinado-6 x finetune_v2 (Etapa 9)
├── threshold_sweep_propio.py          # barrido de umbral de confianza (Etapa 10)
├── reeval_iou045_todas.py             # corrección metodológica de NMS iou (Etapa 10)
├── evaluar_test_yolo.py               # evalúa un best.pt sobre el test agregado
├── evaluar_por_fuente.py              # evalúa desglosado en público vs. propio
├── exportar_tflite.py                 # exporta y cuantiza a TFLite
├── Test_cand_A / Test_metricas_A / Test_all_XML   # Candidato A: ensamble clásico OpenCV
├── models/                            # pesos finales (los únicos binarios versionados)
└── requirements.txt
```

## Datasets (no incluidos en el repo — ver `.gitignore`)

- `md121/` — dataset público (Wageningen), 284 img, clases WF/MR/NC (solo WF se usa).
- `Propio/Tesis.voc/` — dataset propio, 60 img de invernadero real, clase WF.
- `Propio_copypaste*/` — imágenes sintéticas generadas por `copy_paste_augmentation*.py`.
- `yolo_dataset_*/` — datasets ya convertidos a formato YOLO (regenerables con los scripts).
- `yolo_runs/` — corridas de entrenamiento completas (logs + checkpoints intermedios).

Contactar al autor para acceso a los datasets originales.

## Reproducir

```bash
pip install -r requirements.txt
python yolo_entrenamiento.py
```

Configuración de la corrida (fuente de datos, sobremuestreo, copy-paste) se ajusta al inicio de
`yolo_entrenamiento.py`.
