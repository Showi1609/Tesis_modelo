"""Evalua combinado-6 sobre el test_propio con etiquetas CORREGIDAS (Etapa 15), iou=0.45."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt"
DATA_PROPIO = rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"
IMGSZ = 1280
IOU_NMS = 0.45

if __name__ == "__main__":
    modelo = YOLO(MODELO)
    metricas = modelo.val(
        data=DATA_PROPIO,
        split="val",
        imgsz=IMGSZ,
        iou=IOU_NMS,
        device=0,
        workers=0,
        project=rf"{RAIZ}\yolo_runs",
        name="whitefly_combinado6_propio_CORREGIDO",
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print("\n" + "=" * 60)
    print("COMBINADO-6 -- TEST PROPIO CON ETIQUETAS CORREGIDAS (Etapa 15)")
    print("=" * 60)
    print(f"Precision:  {p:.4f}")
    print(f"Recall:     {r:.4f}")
    print(f"mAP50:      {metricas.box.map50:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print("-" * 60)
    print("Antes de la correccion (Etapa 10): P=58.8% R=68.4% mAP50=63.1% F1=0.6325")
    print("=" * 60)
