"""Experimento (continuacion de la Etapa 21): repite combinado-6_v2 exactamente igual pero con
patience=100 en vez de 20. La corrida original paro en la epoca 58 de 100 -- este test confirma
si tambien ahi se estaba perdiendo mejora real por el LR, siguiendo el mismo patron que se
confirmo en fold1_v2 (+6.8 pts de F1 al no cortar prematuramente).

Reutiliza el dataset ya construido (yolo_dataset_combinado6_v2), no hace falta regenerar nada.
Determinista: las primeras 58 epocas deberian reproducir identico a la corrida original."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
RUTA_YAML = rf"{RAIZ}\yolo_dataset_combinado6_v2\data.yaml"
IMGSZ = 1280
BATCH = 4
EPOCAS = 100
PACIENCIA = 100
IOU_EVAL = 0.45

if __name__ == "__main__":
    print("--- Entrenando combinado-6_v2 con patience=100 (experimento, Etapa 21) ---")
    modelo = YOLO("yolov8n.pt")
    modelo.train(
        data=RUTA_YAML, epochs=EPOCAS, imgsz=IMGSZ, batch=BATCH, patience=PACIENCIA,
        device=0, workers=0, degrees=15.0, flipud=0.5,
        project=rf"{RAIZ}\yolo_runs", name="whitefly_yolov8n_combinado6_v2_patience100",
    )

    mejor_pt = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2_patience100\weights\best.pt"
    modelo_eval = YOLO(mejor_pt)

    for nombre_subset, data_yaml in [
        ("propio", rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"),
        ("publico", rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"),
    ]:
        metricas = modelo_eval.val(
            data=data_yaml, split="val", imgsz=IMGSZ, iou=IOU_EVAL, device=0, workers=0, verbose=False,
            project=rf"{RAIZ}\yolo_runs", name=f"whitefly_combinado6_v2_patience100_{nombre_subset}",
        )
        p, r = metricas.box.mp, metricas.box.mr
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"\n=== combinado-6_v2_patience100 -- {nombre_subset} ===")
        print(f"Precision: {p:.4f}  Recall: {r:.4f}  mAP50: {metricas.box.map50:.4f}  F1: {f1:.4f}")

    print("\nPara comparar: combinado-6_v2 original (patience=20, paro en epoca 58): F1 propio=0.6705 F1 publico=0.8556")
