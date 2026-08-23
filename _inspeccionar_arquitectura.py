from ultralytics import YOLO

MODELO = r"C:\Users\jchag\Documents\TESIS\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt"

modelo = YOLO(MODELO)
print(modelo.model)
print("\n\n=== INFO ===")
modelo.info(detailed=True, verbose=True)
