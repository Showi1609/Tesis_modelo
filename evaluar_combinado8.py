"""Evalua combinado-8 (self copy-paste) con iou=0.45, misma metodologia que Etapa 10,
para comparar directo contra combinado-6 y combinado-7."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-8\weights\best.pt"
DATA_AGREGADO = rf"{RAIZ}\yolo_dataset_combinado8\data.yaml"
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
    agregado = evaluar(DATA_AGREGADO, "whitefly_combinado8_agregado")
    publico = evaluar(DATA_PUBLICO, "whitefly_combinado8_publico")
    propio = evaluar(DATA_PROPIO, "whitefly_combinado8_propio")
    gap = publico["F1"] - propio["F1"]

    print("\n" + "=" * 80)
    print("COMBINADO-8 (self copy-paste) vs COMBINADO-6 vs COMBINADO-7 -- todos iou=0.45")
    print("=" * 80)
    print(f"{'':>12} | {'F1 agregado':>12} | {'F1 publico':>11} | {'F1 propio':>10} | {'gap':>8}")
    print(f"{'combinado-6':>12} | {0.7445:>12.4f} | {0.8149:>11.4f} | {0.6325:>10.4f} | {0.1824:>8.4f}")
    print(f"{'combinado-7':>12} | {0.6913:>12.4f} | {0.8029:>11.4f} | {0.5123:>10.4f} | {0.2906:>8.4f}")
    print(f"{'combinado-8':>12} | {agregado['F1']:>12.4f} | {publico['F1']:>11.4f} | {propio['F1']:>10.4f} | {gap:>8.4f}")
    print("=" * 80)
