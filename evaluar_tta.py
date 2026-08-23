"""Evalua combinado-6_v2 y combinado-8_v2 con Test-Time Augmentation (augment=True) sobre propio
y publico, mismo iou=0.45 de siempre, para comparar contra el baseline sin TTA."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
IMGSZ = 1280
IOU_NMS = 0.45

MODELOS = {
    "combinado-6_v2": rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\weights\best.pt",
    "combinado-8_v2": rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-8_v2\weights\best.pt",
}

SUBSETS = {
    "propio": rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml",
    "publico": rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml",
}

if __name__ == "__main__":
    for nombre_modelo, ruta_modelo in MODELOS.items():
        modelo = YOLO(ruta_modelo)
        for nombre_subset, data_yaml in SUBSETS.items():
            metricas = modelo.val(
                data=data_yaml, split="val", imgsz=IMGSZ, iou=IOU_NMS, device=0, workers=0,
                verbose=False, augment=True,
                project=rf"{RAIZ}\yolo_runs", name=f"whitefly_{nombre_modelo}_tta_{nombre_subset}",
            )
            p, r = metricas.box.mp, metricas.box.mr
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print(f"=== {nombre_modelo} TTA -- {nombre_subset} ===")
            print(f"Precision: {p:.4f}  Recall: {r:.4f}  mAP50: {metricas.box.map50:.4f}  F1: {f1:.4f}")
