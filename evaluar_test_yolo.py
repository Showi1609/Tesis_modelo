from ultralytics import YOLO

MODELO = r"C:\Users\jchag\Documents\TESIS\yolo_runs\whitefly_yolov8n_finetune_propio_v2\weights\best.pt"
DATA_YAML = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_combinado\data.yaml"

if __name__ == "__main__":
    modelo = YOLO(MODELO)
    metricas = modelo.val(
        data=DATA_YAML,
        split="test",
        imgsz=1280,
        device=0,
        workers=0,
        project=r"C:\Users\jchag\Documents\TESIS\yolo_runs",
        name="whitefly_yolov8n_test_finetune_v2",
    )

    print("\n" + "=" * 50)
    print("RESULTADOS FINALES SOBRE EL CONJUNTO DE TEST")
    print("=" * 50)
    print(f"Precision (mp):     {metricas.box.mp:.4f}")
    print(f"Recall (mr):        {metricas.box.mr:.4f}")
    print(f"mAP50:              {metricas.box.map50:.4f}")
    print(f"mAP50-95:           {metricas.box.map:.4f}")
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"F1-Score:           {f1:.4f}")
    print("=" * 50)
