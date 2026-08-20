import os
import shutil
from ultralytics import YOLO

# Fine-tuning v2: continuar el entrenamiento del mejor modelo (combinado + copy-paste)
# usando el propio AMPLIADO (42 reales + 126 sintéticas copy-paste = 168), para especializarlo
# a las condiciones del invernadero propio sin quedarse corto de datos como en el v1 (solo 42).
BASE = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_combinado"
MODELO_BASE = r"C:\Users\jchag\Documents\TESIS\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt"
DIR_SALIDA = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_finetune_propio_v2"
PREFIJOS_TRAIN = ("propio_", "cp_")  # reales + sintéticas copy-paste
PREFIJOS_VAL = ("propio_",)          # val se queda solo con imágenes reales (nunca sintéticas)

EPOCAS_FINETUNE = 40
PACIENCIA_FINETUNE = 15
LR_FINETUNE = 0.001  # más bajo que el default (0.01): ya no aprende desde cero, solo se ajusta


def construir_subset(prefijos, split_origen, split_destino):
    """Copia solo las imágenes/labels que empiecen con alguno de `prefijos` desde split_origen
    (train/val en el dataset combinado) hacia split_destino en el dataset de fine-tuning."""
    dir_img_dst = os.path.join(DIR_SALIDA, "images", split_destino)
    dir_lbl_dst = os.path.join(DIR_SALIDA, "labels", split_destino)
    os.makedirs(dir_img_dst, exist_ok=True)
    os.makedirs(dir_lbl_dst, exist_ok=True)

    dir_img_src = os.path.join(BASE, "images", split_origen)
    dir_lbl_src = os.path.join(BASE, "labels", split_origen)

    n = 0
    for nombre_archivo in os.listdir(dir_img_src):
        if not nombre_archivo.startswith(prefijos):
            continue
        shutil.copy2(os.path.join(dir_img_src, nombre_archivo), os.path.join(dir_img_dst, nombre_archivo))
        base = os.path.splitext(nombre_archivo)[0]
        ruta_lbl_src = os.path.join(dir_lbl_src, base + ".txt")
        ruta_lbl_dst = os.path.join(dir_lbl_dst, base + ".txt")
        if os.path.exists(ruta_lbl_src):
            shutil.copy2(ruta_lbl_src, ruta_lbl_dst)
        else:
            open(ruta_lbl_dst, "w").close()
        n += 1
    return n


if __name__ == "__main__":
    for sub in ("images", "labels"):
        ruta = os.path.join(DIR_SALIDA, sub)
        if os.path.isdir(ruta):
            shutil.rmtree(ruta)

    n_train = construir_subset(PREFIJOS_TRAIN, "train", "train")
    n_val = construir_subset(PREFIJOS_VAL, "val", "val")
    print(f"Fine-tuning train: {n_train} imágenes (reales + copy-paste)")
    print(f"Fine-tuning val:   {n_val} imágenes propias reales")
    print("(el test_propio de la evaluación por fuente NO se toca: queda como held-out para comparar antes/después)")

    ruta_yaml = os.path.join(DIR_SALIDA, "data.yaml")
    with open(ruta_yaml, "w") as f:
        f.write(f"path: {DIR_SALIDA}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: WF\n")

    print(f"\n--- Fine-tuning desde {MODELO_BASE} ---")
    modelo = YOLO(MODELO_BASE)
    modelo.train(
        data=ruta_yaml,
        epochs=EPOCAS_FINETUNE,
        imgsz=1280,
        batch=4,
        patience=PACIENCIA_FINETUNE,
        lr0=LR_FINETUNE,
        optimizer="AdamW",  # con optimizer="auto" Ultralytics ignora lr0 y elige su propio LR; para fine-tuning suave hay que fijarlo
        device=0,
        workers=0,
        degrees=15.0,
        flipud=0.5,
        project=r"C:\Users\jchag\Documents\TESIS\yolo_runs",
        name="whitefly_yolov8n_finetune_propio_v2",
    )
