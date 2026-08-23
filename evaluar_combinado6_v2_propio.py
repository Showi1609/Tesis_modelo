"""Evalua combinado6_v2 (reentrenado con etiquetas propias corregidas, Etapa 19) sobre el
test_propio oficial (9 img, ya reconstruido con las etiquetas mas recientes), iou=0.45."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\weights\best.pt"
DATA_PROPIO = rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"
IMGSZ = 1280
IOU_NMS = 0.45

if __name__ == "__main__":
    modelo = YOLO(MODELO)
    metricas = modelo.val(
        data=DATA_PROPIO, split="val", imgsz=IMGSZ, iou=IOU_NMS, device=0, workers=0,
        project=rf"{RAIZ}\yolo_runs", name="whitefly_combinado6_v2_propio",
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print("\n" + "=" * 60)
    print("COMBINADO-6_V2 -- TEST PROPIO CON ETIQUETAS RE-CORREGIDAS (Etapa 19)")
    print("=" * 60)
    print(f"Precision:  {p:.4f}")
    print(f"Recall:     {r:.4f}")
    print(f"mAP50:      {metricas.box.map50:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print("-" * 60)
    print("Referencia -- combinado-6 original (Etapa 15): P=80.6% R=71.5% F1=0.7578")
    print("Referencia -- 5-fold CV robusto (Etapa 18, datos viejos): F1=0.4960+-0.0624")
    print("=" * 60)
