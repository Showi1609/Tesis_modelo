"""Reevalua combinado6_v2 (propio y publico) con iou=0.1 -- mismo criterio permisivo usado para
Candidato A, para una comparacion final consistente en ambos sentidos (estricto y permisivo)."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\weights\best.pt"
IMGSZ = 1280
IOU_NMS = 0.1

modelo = YOLO(MODELO)

for nombre_subset, data_yaml in [
    ("propio", rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"),
    ("publico", rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"),
]:
    metricas = modelo.val(
        data=data_yaml, split="val", imgsz=IMGSZ, iou=IOU_NMS, device=0, workers=0,
        project=rf"{RAIZ}\yolo_runs", name=f"whitefly_combinado6_v2_{nombre_subset}_iou01",
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"\n=== combinado6_v2 -- {nombre_subset} -- iou=0.1 ===")
    print(f"Precision: {p:.4f}  Recall: {r:.4f}  mAP50: {metricas.box.map50:.4f}  F1: {f1:.4f}")
