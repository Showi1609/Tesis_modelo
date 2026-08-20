"""
Reevalua combinado-5 (sin sobremuestreo), combinado-3 (con sobremuestreo) y combinado-6
(copy-paste) con iou=0.45 -- el NMS IoU que realmente corre en la app Android (Kotlin,
funcion nms() con umbralIou=0.45f) -- en vez del default de Ultralytics (iou=0.7) que se uso
para todas las cifras "oficiales" de la bitacora hasta ahora.

No se reentrena nada; son los mismos .pt de siempre. conf se deja en su valor por defecto
(None -> Ultralytics elige internamente el punto de la curva P/R que maximiza F1 por clase),
igual que en todas las evaluaciones anteriores, para que el unico cambio real entre esta
tabla y la de la bitacora sea el iou -- una comparacion limpia, sin mezclar dos variables.

Cubre exactamente lo que se promete actualizar en el artifact:
  - "Comparacion final" (Etapa 3): clasico / sin sobremuestreo / con sobremuestreo
  - "Desglose por fuente" del modelo con sobremuestreo (Etapa 4)
  - "Evolucion completa" (Etapa 6): agrega copy-paste
  - "Desglose por fuente" del modelo con copy-paste (Etapa 6)

Etapa 5 (fine-tuning) y Etapa 7 (descartada) no se tocan en este script -- se documentaran
en el artifact como pendientes de reevaluar si hace falta.
"""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DATA_AGREGADO = rf"{RAIZ}\yolo_dataset_combinado\data.yaml"
DATA_PUBLICO = rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"
DATA_PROPIO = rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"
IMGSZ = 1280
IOU_NMS = 0.45  # igual al NMS manual en Kotlin

MODELOS = {
    "sin_sobremuestreo": rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-5\weights\best.pt",
    "con_sobremuestreo":  rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-3\weights\best.pt",
    "copy_paste":         rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt",
}


def evaluar(modelo_path, data_yaml, nombre_run):
    modelo = YOLO(modelo_path)
    metricas = modelo.val(
        data=data_yaml,
        split="test" if data_yaml == DATA_AGREGADO else "val",
        imgsz=IMGSZ,
        iou=IOU_NMS,
        device=0,
        workers=0,
        project=rf"{RAIZ}\yolo_runs",
        name=nombre_run,
        verbose=False,
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"P": p, "R": r, "mAP50": metricas.box.map50, "mAP50-95": metricas.box.map, "F1": f1}


if __name__ == "__main__":
    resultados = {}
    for nombre, ruta in MODELOS.items():
        print(f"\n{'=' * 60}\n{nombre}\n{'=' * 60}")
        agregado = evaluar(ruta, DATA_AGREGADO, f"whitefly_iou045_{nombre}_agregado")
        publico = evaluar(ruta, DATA_PUBLICO, f"whitefly_iou045_{nombre}_publico")
        propio = evaluar(ruta, DATA_PROPIO, f"whitefly_iou045_{nombre}_propio")
        resultados[nombre] = {"agregado": agregado, "publico": publico, "propio": propio}
        print(f"  agregado F1={agregado['F1']:.4f} P={agregado['P']:.4f} R={agregado['R']:.4f} mAP50={agregado['mAP50']:.4f}")
        print(f"  publico  F1={publico['F1']:.4f} P={publico['P']:.4f} R={publico['R']:.4f} mAP50={publico['mAP50']:.4f}")
        print(f"  propio   F1={propio['F1']:.4f} P={propio['P']:.4f} R={propio['R']:.4f} mAP50={propio['mAP50']:.4f}")

    print("\n" + "=" * 100)
    print("COMPARACION FINAL (test agregado, 53 img, 814 cajas) -- iou=0.45")
    print(f"{'':>20} | {'Precision':>10} | {'Recall':>10} | {'mAP50':>10} | {'F1':>8}")
    for nombre in MODELOS:
        a = resultados[nombre]["agregado"]
        print(f"{nombre:>20} | {a['P']:>10.4f} | {a['R']:>10.4f} | {a['mAP50']:>10.4f} | {a['F1']:>8.4f}")

    print("\n" + "=" * 100)
    print("DESGLOSE POR FUENTE -- iou=0.45")
    for nombre in MODELOS:
        pub, pro = resultados[nombre]["publico"], resultados[nombre]["propio"]
        gap = pub["F1"] - pro["F1"]
        print(f"\n[{nombre}]")
        print(f"  publico F1={pub['F1']:.4f}  propio F1={pro['F1']:.4f}  gap={gap:.4f}")
    print("=" * 100)
