"""Experimento: repite fold1_v2 (Etapa 19b) exactamente igual pero con patience=100 (efectivamente
sin early-stopping dentro de 100 epocas), para ver si el corte en la epoca 31 (patience=20) dejo
mejoras sin capturar por el LR todavia alto. Como todo es determinista (mismas semillas), las
primeras ~31 epocas deberian reproducir identico al run original -- este solo agrega lo que sigue.

Reutiliza el dataset ya construido (yolo_dataset_fold1_v2), no hace falta regenerar nada."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
RUTA_YAML = rf"{RAIZ}\yolo_dataset_fold1_v2\data.yaml"
RUTA_YAML_TEST_PROPIO = rf"{RAIZ}\yolo_dataset_fold1_test_propio_v2\data.yaml"
IMGSZ = 1280
BATCH = 4
EPOCAS = 100
PACIENCIA = 100  # = EPOCAS, para que nunca dispare antes de terminar las 100
IOU_EVAL = 0.45

if __name__ == "__main__":
    print("--- Entrenando fold1_v2 con patience=100 (experimento) ---")
    modelo = YOLO("yolov8n.pt")
    modelo.train(
        data=RUTA_YAML, epochs=EPOCAS, imgsz=IMGSZ, batch=BATCH, patience=PACIENCIA,
        device=0, workers=0, degrees=15.0, flipud=0.5,
        project=rf"{RAIZ}\yolo_runs", name="whitefly_yolov8n_fold1_v2_patience100",
    )

    mejor_pt = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_fold1_v2_patience100\weights\best.pt"
    print("\n--- Evaluando sobre el test propio held-out del fold1 ---")
    modelo_eval = YOLO(mejor_pt)
    metricas = modelo_eval.val(
        data=RUTA_YAML_TEST_PROPIO, split="test", imgsz=IMGSZ, iou=IOU_EVAL, device=0, workers=0,
        project=rf"{RAIZ}\yolo_runs", name="whitefly_fold1_v2_patience100_eval_propio",
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print("\n" + "=" * 60)
    print(f"fold1_v2 patience=100: P={p:.4f} R={r:.4f} F1={f1:.4f}")
    print("Para comparar: fold1_v2 original (patience=20, paro en epoca 31): F1=0.5465")
    print("=" * 60)
