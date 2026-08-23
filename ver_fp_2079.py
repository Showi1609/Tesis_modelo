from PIL import Image, ImageDraw
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_fold5\weights\best.pt"
IMG = rf"{RAIZ}\yolo_dataset_fold5_test_propio\images\test"

import glob, os
ruta_img = [p for p in glob.glob(os.path.join(IMG, "*.jpg")) if "2079" in p][0]

modelo = YOLO(MODELO)
resultado = modelo.predict(source=ruta_img, imgsz=1280, conf=0.25, iou=0.45, device=0, verbose=False)[0]

with Image.open(ruta_img) as img:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in resultado.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        color = (255, 0, 0) if conf >= 0.5 else (255, 165, 0)
        draw.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=color, width=3)
    img.save(rf"{RAIZ}\diagnostico_fp_fold1\_2079_overview.jpg", "JPEG", quality=90)
    print(f"Total detecciones: {len(resultado.boxes)}")
