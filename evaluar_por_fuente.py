import os
import shutil
from ultralytics import YOLO

BASE = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_combinado"
MODELO = r"C:\Users\jchag\Documents\TESIS\yolo_runs\whitefly_yolov8n_finetune_propio_v2\weights\best.pt"
DIR_SALIDA = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_desglose"


def construir_subset(prefijo, nombre_subset):
    """Copia del test combinado solo las imágenes/labels cuyo nombre empieza con `prefijo`."""
    dir_img_dst = os.path.join(DIR_SALIDA, "images", nombre_subset)
    dir_lbl_dst = os.path.join(DIR_SALIDA, "labels", nombre_subset)
    if os.path.isdir(dir_img_dst):
        shutil.rmtree(dir_img_dst)
    if os.path.isdir(dir_lbl_dst):
        shutil.rmtree(dir_lbl_dst)
    os.makedirs(dir_img_dst, exist_ok=True)
    os.makedirs(dir_lbl_dst, exist_ok=True)

    dir_img_src = os.path.join(BASE, "images", "test")
    dir_lbl_src = os.path.join(BASE, "labels", "test")

    n_imagenes, n_cajas = 0, 0
    for nombre_archivo in os.listdir(dir_img_src):
        if not nombre_archivo.startswith(prefijo):
            continue
        shutil.copy2(os.path.join(dir_img_src, nombre_archivo), os.path.join(dir_img_dst, nombre_archivo))
        base = os.path.splitext(nombre_archivo)[0]
        ruta_lbl_src = os.path.join(dir_lbl_src, base + ".txt")
        ruta_lbl_dst = os.path.join(dir_lbl_dst, base + ".txt")
        if os.path.exists(ruta_lbl_src):
            shutil.copy2(ruta_lbl_src, ruta_lbl_dst)
            with open(ruta_lbl_src) as f:
                n_cajas += len([linea for linea in f if linea.strip()])
        else:
            open(ruta_lbl_dst, "w").close()
        n_imagenes += 1

    return n_imagenes, n_cajas


def escribir_yaml(nombre_subset):
    ruta_yaml = os.path.join(DIR_SALIDA, f"data_{nombre_subset}.yaml")
    with open(ruta_yaml, "w") as f:
        f.write(f"path: {DIR_SALIDA}\n")
        f.write(f"train: images/{nombre_subset}\n")
        f.write(f"val: images/{nombre_subset}\n")
        f.write("names:\n  0: WF\n")
    return ruta_yaml


if __name__ == "__main__":
    n_pub, cajas_pub = construir_subset("md121_", "test_publico")
    n_pro, cajas_pro = construir_subset("propio_", "test_propio")
    print(f"[publico] {n_pub} imagenes, {cajas_pub} cajas WF")
    print(f"[propio]  {n_pro} imagenes, {cajas_pro} cajas WF")

    modelo = YOLO(MODELO)
    resultados = {}

    for nombre_subset in ["test_publico", "test_propio"]:
        ruta_yaml = escribir_yaml(nombre_subset)
        print(f"\n=== Evaluando subset: {nombre_subset} ===")
        metricas = modelo.val(
            data=ruta_yaml,
            split="val",
            imgsz=1280,
            device=0,
            workers=0,
            project=r"C:\Users\jchag\Documents\TESIS\yolo_runs",
            name=f"whitefly_yolov8n_desglose_{nombre_subset}",
        )
        p, r = metricas.box.mp, metricas.box.mr
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        resultados[nombre_subset] = (p, r, metricas.box.map50, metricas.box.map, f1)

    print("\n" + "=" * 60)
    print("DESGLOSE POR FUENTE - CONJUNTO DE TEST")
    print("=" * 60)
    for nombre_subset, (p, r, map50, map5095, f1) in resultados.items():
        print(f"\n[{nombre_subset}]")
        print(f"  Precision:  {p:.4f}")
        print(f"  Recall:     {r:.4f}")
        print(f"  mAP50:      {map50:.4f}")
        print(f"  mAP50-95:   {map5095:.4f}")
        print(f"  F1-Score:   {f1:.4f}")
    print("=" * 60)
