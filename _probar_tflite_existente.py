from ultralytics import YOLO

RUTA_TFLITE = r"C:\Users\jchag\Documents\TESIS\yolo_runs\whitefly_yolov8n_combinado-6\weights\best_saved_model\best_integer_quant.tflite"
DATA_YAML = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_combinado\data.yaml"

if __name__ == "__main__":
    modelo_tflite = YOLO(RUTA_TFLITE, task="detect")
    metricas = modelo_tflite.val(
        data=DATA_YAML,
        split="test",
        imgsz=1280,
        project=r"C:\Users\jchag\Documents\TESIS\yolo_runs",
        name="whitefly_yolov8n_tflite_int8_test",
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print("\n" + "=" * 50)
    print(f"Precision: {p:.4f}  Recall: {r:.4f}  mAP50: {metricas.box.map50:.4f}  mAP50-95: {metricas.box.map:.4f}  F1: {f1:.4f}")
    print("=" * 50)
