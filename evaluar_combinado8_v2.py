"""Evalua combinado-8_v2 (Etapa 12 repetida con etiquetas corregidas) sobre propio y publico,
iou=0.45, mismo criterio que el resto de la tesis."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-8_v2\weights\best.pt"
IMGSZ = 1280
IOU_NMS = 0.45

modelo = YOLO(MODELO)

for nombre_subset, data_yaml in [
    ("propio", rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"),
    ("publico", rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"),
]:
    metricas = modelo.val(
        data=data_yaml, split="val", imgsz=IMGSZ, iou=IOU_NMS, device=0, workers=0, verbose=False,
        project=rf"{RAIZ}\yolo_runs", name=f"whitefly_combinado8_v2_{nombre_subset}",
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"=== combinado-8_v2 -- {nombre_subset} ===")
    print(f"Precision: {p:.4f}  Recall: {r:.4f}  mAP50: {metricas.box.map50:.4f}  F1: {f1:.4f}")
