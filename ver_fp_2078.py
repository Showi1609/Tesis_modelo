"""Dibuja todas las detecciones (FP) del modelo fold1 sobre la imagen completa propio_2078,
para ver el patron espacial -- si estan dispersas por toda la trampa (sugiere ruido de textura)
o concentradas en una zona (sugiere algo especifico en esa parte de la imagen)."""

from PIL import Image, ImageDraw
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_fold1\weights\best.pt"
IMG = rf"{RAIZ}\yolo_dataset_fold1_test_propio\images\test\propio_2078_jpg.rf.ZjLW3cBf5Y4t8Oh6DOze.jpg"

modelo = YOLO(MODELO)
resultado = modelo.predict(source=IMG, imgsz=1280, conf=0.25, iou=0.45, device="cpu", verbose=False)[0]

with Image.open(IMG) as img:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in resultado.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        color = (255, 0, 0) if conf >= 0.5 else (255, 165, 0)
        draw.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=color, width=3)
    img.save(rf"{RAIZ}\diagnostico_fp_fold1\_2078_overview.jpg", "JPEG", quality=90)
    print(f"Total detecciones: {len(resultado.boxes)}")
    print("Guardado en _2078_overview.jpg")
