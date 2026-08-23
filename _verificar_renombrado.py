from ultralytics import YOLO

DIR = r"C:\Users\jchag\Documents\TESIS\yolo_runs\whitefly_yolov8n_combinado-6\weights\best_saved_model"
DATA_YAML = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_combinado\data.yaml"

VARIANTES = ["best_int8.tflite", "best_full_integer_quant.tflite", "best_integer_quant.tflite"]

if __name__ == "__main__":
    for nombre in VARIANTES:
        print(f"\n=== {nombre} ===")
        modelo = YOLO(f"{DIR}\\{nombre}", task="detect")
        metricas = modelo.val(
            data=DATA_YAML, split="test", imgsz=1280,
            project=r"C:\Users\jchag\Documents\TESIS\yolo_runs",
            name=f"whitefly_verif_{nombre.replace('.tflite','')}",
        )
        p, r = metricas.box.mp, metricas.box.mr
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{nombre}: P={p:.4f} R={r:.4f} mAP50={metricas.box.map50:.4f} F1={f1:.4f}")
