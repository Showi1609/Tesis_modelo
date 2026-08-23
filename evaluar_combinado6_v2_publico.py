"""Evalua combinado6_v2 sobre test_publico (44 img, sin cambios), iou=0.45 -- para comparar
domain gap contra combinado-6 original."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\weights\best.pt"
DATA_PUBLICO = rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"
IMGSZ = 1280
IOU_NMS = 0.45

if __name__ == "__main__":
    modelo = YOLO(MODELO)
    metricas = modelo.val(
        data=DATA_PUBLICO, split="val", imgsz=IMGSZ, iou=IOU_NMS, device=0, workers=0,
        project=rf"{RAIZ}\yolo_runs", name="whitefly_combinado6_v2_publico",
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print("\n" + "=" * 60)
    print("COMBINADO-6_V2 -- TEST PUBLICO")
    print("=" * 60)
    print(f"Precision:  {p:.4f}")
    print(f"Recall:     {r:.4f}")
    print(f"mAP50:      {metricas.box.map50:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print("=" * 60)
