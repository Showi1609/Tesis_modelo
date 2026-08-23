from ultralytics import YOLO

DIR = r"C:\Users\jchag\Documents\TESIS\yolo_runs\whitefly_yolov8n_combinado-6\weights\best_saved_model"
DATA_YAML = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_combinado\data.yaml"

VARIANTES = [
    "best_float32.tflite",
    "best_float16.tflite",
    "best_dynamic_range_quant.tflite",
    "best_integer_quant.tflite",
]

if __name__ == "__main__":
    resultados = {}
    for nombre in VARIANTES:
        ruta = f"{DIR}\\{nombre}"
        print(f"\n=== Evaluando {nombre} ===")
        modelo = YOLO(ruta, task="detect")
        metricas = modelo.val(
            data=DATA_YAML,
            split="test",
            imgsz=1280,
            project=r"C:\Users\jchag\Documents\TESIS\yolo_runs",
            name=f"whitefly_tflite_compare_{nombre.replace('.tflite', '')}",
        )
        p, r = metricas.box.mp, metricas.box.mr
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        resultados[nombre] = (p, r, metricas.box.map50, metricas.box.map, f1)

    print("\n" + "=" * 70)
    print("COMPARACION DE VARIANTES TFLITE - TEST SET")
    print("=" * 70)
    for nombre, (p, r, map50, map5095, f1) in resultados.items():
        print(f"{nombre:35s}  P={p:.4f}  R={r:.4f}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}  F1={f1:.4f}")
    print("=" * 70)
