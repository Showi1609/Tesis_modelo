"""Exporta combinado-6_v2 a TFLite con la misma metodologia que combinado-6 (Etapa previa):
int8 dynamic-range (pesos en INT8 sin calibracion, la variante "best_int8.tflite") -- la
static/integer_quant con calibracion ya se probo que colapsa el modelo, no se repite ese intento."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO_BASE = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\weights\best.pt"
DATA_YAML = rf"{RAIZ}\yolo_dataset_combinado6_v2\data.yaml"
IMGSZ = 1280

if __name__ == "__main__":
    modelo = YOLO(MODELO_BASE)
    ruta_tflite = modelo.export(format="tflite", int8=True, data=DATA_YAML, imgsz=IMGSZ)
    print(f"\nModelo TFLite generado en: {ruta_tflite}")
