"""Evalua combinado-10 (combinado-6 + 24 negativos duros) contra el test propio corregido
(Etapa 15), iou=0.45, comparando contra combinado-6."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-10\weights\best.pt"
DATA_AGREGADO = rf"{RAIZ}\yolo_dataset_combinado10\data.yaml"
DATA_PUBLICO = rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"
DATA_PROPIO = rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"
IMGSZ = 1280
IOU_NMS = 0.45


def evaluar(data_yaml, nombre_run):
    modelo = YOLO(MODELO)
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
    return {"P": p, "R": r, "mAP50": metricas.box.map50, "F1": f1}


if __name__ == "__main__":
    agregado = evaluar(DATA_AGREGADO, "whitefly_combinado10_agregado")
    publico = evaluar(DATA_PUBLICO, "whitefly_combinado10_publico")
    propio = evaluar(DATA_PROPIO, "whitefly_combinado10_propio_CORREGIDO")

    print("\n" + "=" * 70)
    print("COMBINADO-10 (+ 24 negativos duros) vs COMBINADO-6 -- etiquetas corregidas, iou=0.45")
    print("=" * 70)
    print(f"{'':>14} | {'F1 agregado':>12} | {'F1 publico':>11} | {'F1 propio':>10}")
    print(f"{'combinado-6':>14} | {'—':>12} | {0.8149:>11.4f} | {0.7578:>10.4f}")
    print(f"{'combinado-10':>14} | {agregado['F1']:>12.4f} | {publico['F1']:>11.4f} | {propio['F1']:>10.4f}")
    print("=" * 70)
    print(f"\ncombinado-10 propio: P={propio['P']:.4f} R={propio['R']:.4f} F1={propio['F1']:.4f}")
    print(f"combinado-10 publico: P={publico['P']:.4f} R={publico['R']:.4f} F1={publico['F1']:.4f}")
