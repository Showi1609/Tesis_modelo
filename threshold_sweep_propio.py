"""
Barrido de umbral de confianza sobre combinado-6/best.pt (el modelo YA en produccion, sin
tocar pesos) para encontrar el conf optimo para F1 en el subset propio (9 img, condiciones
reales de invernadero) - Pendiente #3 de la bitacora.

Importante: las evaluaciones anteriores (evaluar_test_yolo.py, evaluar_por_fuente.py,
weight_soup_propio.py) llamaron a model.val() SIN pasar conf/iou explicitos. Ultralytics usa
por defecto conf~0.001 en validacion (para poder calcular mAP integrando sobre toda la curva
precision-recall) y reporta P/R en el punto de la curva que maximiza F1 por clase - NO
necesariamente el conf=0.25 que la app Android usa hoy (ver Etapa 8/pendiente 3 de la
bitacora). Osea que el F1=0.619 en propio reportado hasta ahora es optimista respecto a lo que
la app realmente entrega con su umbral fijo actual.

Este script si fija conf e iou explicitos (iou=0.45 para igualar el NMS que corre en Kotlin) y
barre varios conf, evaluando por separado en propio/publico/agregado, para:
  1. ver que F1 da realmente el conf=0.25 actual de la app en propio
  2. encontrar si hay un conf que mejore F1 propio manteniendo lo demas razonable

No requiere reentrenar nada; el .pt no cambia.
"""

import os

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt"
DATA_AGREGADO = rf"{RAIZ}\yolo_dataset_combinado\data.yaml"
DATA_PUBLICO = rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"
DATA_PROPIO = rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"
IMGSZ = 1280
IOU_NMS = 0.45  # igual al NMS manual implementado en Kotlin

CONFS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def evaluar(data_yaml, conf, nombre_run):
    modelo = YOLO(MODELO)
    metricas = modelo.val(
        data=data_yaml,
        split="test" if data_yaml == DATA_AGREGADO else "val",
        imgsz=IMGSZ,
        conf=conf,
        iou=IOU_NMS,
        device=0,
        workers=0,
        project=rf"{RAIZ}\yolo_runs",
        name=nombre_run,
        verbose=False,
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"P": p, "R": r, "mAP50": metricas.box.map50, "F1": f1}


if __name__ == "__main__":
    filas = []
    for conf in CONFS:
        tag = f"c{int(conf*100):03d}"
        print(f"\n{'=' * 60}\nconf={conf}\n{'=' * 60}")
        agregado = evaluar(DATA_AGREGADO, conf, f"whitefly_thresh_{tag}_agregado")
        publico = evaluar(DATA_PUBLICO, conf, f"whitefly_thresh_{tag}_publico")
        propio = evaluar(DATA_PROPIO, conf, f"whitefly_thresh_{tag}_propio")
        filas.append((conf, agregado, publico, propio))
        print(f"  agregado F1={agregado['F1']:.4f}  publico F1={publico['F1']:.4f}  propio F1={propio['F1']:.4f}")

    print("\n" + "=" * 110)
    print(f"{'conf':>6} | {'P prop':>7} | {'R prop':>7} | {'F1 prop':>8} | {'F1 pub':>7} | {'F1 agreg':>9}")
    print("-" * 110)
    mejor_propio = max(filas, key=lambda f: f[3]["F1"])
    for conf, agregado, publico, propio in filas:
        marca = "  <-- conf actual de la app" if abs(conf - 0.25) < 1e-6 else ""
        marca = "  <-- mejor F1 propio" if conf == mejor_propio[0] else marca
        print(f"{conf:>6.2f} | {propio['P']:>7.4f} | {propio['R']:>7.4f} | {propio['F1']:>8.4f} | "
              f"{publico['F1']:>7.4f} | {agregado['F1']:>9.4f}{marca}")
    print("=" * 110)
