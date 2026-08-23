"""Fine-tuning corto desde el checkpoint de combinado-6_v2 (paro en epoca 58/100 por
early stopping, patience=20). Fase de pulido: 25 epocas mas, LR bajo y fijo, sin mosaico.
Mismo patron que finetune_combinado8_v2.py (que dio F1 propio~sin cambio, publico +2.2,
no adoptado). Se prueba aca porque combinado-6_v2 paro mas temprano (58 vs 100 epocas),
en teoria con mas margen para pulir."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
CHECKPOINT_BASE = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\weights\best.pt"
RUTA_YAML = rf"{RAIZ}\yolo_dataset_combinado6_v2\data.yaml"
IMGSZ = 1280
BATCH = 4
EPOCAS_FINETUNE = 25
LR0_FINETUNE = 0.0002
IOU_EVAL = 0.45

if __name__ == "__main__":
    print("--- Fine-tuning de combinado-6_v2 (25 epocas, LR bajo, sin mosaico) ---")
    modelo = YOLO(CHECKPOINT_BASE)
    modelo.train(
        data=RUTA_YAML, epochs=EPOCAS_FINETUNE, imgsz=IMGSZ, batch=BATCH, patience=EPOCAS_FINETUNE,
        device=0, workers=0, degrees=15.0, flipud=0.5, mosaic=0.0,
        optimizer="AdamW", lr0=LR0_FINETUNE, cos_lr=True,
        project=rf"{RAIZ}\yolo_runs", name="whitefly_yolov8n_combinado6_v2_finetune",
    )

    mejor_pt = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2_finetune\weights\best.pt"
    modelo_eval = YOLO(mejor_pt)

    for nombre_subset, data_yaml in [
        ("propio", rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"),
        ("publico", rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"),
    ]:
        metricas = modelo_eval.val(
            data=data_yaml, split="val", imgsz=IMGSZ, iou=IOU_EVAL, device=0, workers=0, verbose=False,
            project=rf"{RAIZ}\yolo_runs", name=f"whitefly_combinado6_v2_finetune_{nombre_subset}",
        )
        p, r = metricas.box.mp, metricas.box.mr
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"\n=== combinado-6_v2_finetune -- {nombre_subset} ===")
        print(f"Precision: {p:.4f}  Recall: {r:.4f}  mAP50: {metricas.box.map50:.4f}  F1: {f1:.4f}")

    print("\nPara comparar: combinado-6_v2 sin fine-tune: F1 propio=0.6705 F1 publico=0.8556")
