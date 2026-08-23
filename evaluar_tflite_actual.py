"""Evalua el best_int8.tflite ACTUAL (combinado-6 original, el que esta desplegado en la app)
sobre los mismos test sets propio/publico, para comparar contra combinado6_v2 en igualdad de
condiciones (mismo formato TFLite INT8, mismos test sets, mismo iou)."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO_TFLITE = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-6\weights\best_saved_model\best_int8.tflite"
IMGSZ = 1280
IOU_NMS = 0.45

modelo = YOLO(MODELO_TFLITE)

for nombre_subset, data_yaml in [
    ("propio", rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"),
    ("publico", rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"),
]:
    metricas = modelo.val(
        data=data_yaml, split="val", imgsz=IMGSZ, iou=IOU_NMS, verbose=False,
        project=rf"{RAIZ}\yolo_runs", name=f"whitefly_combinado6_actual_tflite_{nombre_subset}",
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"=== combinado-6 (ACTUAL en app) INT8 TFLite -- {nombre_subset} ===")
    print(f"Precision: {p:.4f}  Recall: {r:.4f}  mAP50: {metricas.box.map50:.4f}  F1: {f1:.4f}")
