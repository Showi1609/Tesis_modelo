"""Reevalua combinado-7, 8, 9 y alpha=0.50 (Etapa 9) contra el test_propio con etiquetas
corregidas (Etapa 15), iou=0.45 -- para tener una tabla comparativa justa contra el nuevo
F1 propio de combinado-6 (0.7578)."""

from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DATA_PROPIO = rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"
IMGSZ = 1280
IOU_NMS = 0.45

MODELOS = {
    "combinado-6":   rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt",
    "combinado-7":   rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-7-2\weights\best.pt",
    "combinado-8":   rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-8\weights\best.pt",
    "combinado-9":   rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-9\weights\best.pt",
    "alpha=0.50":    rf"{RAIZ}\yolo_runs\whitefly_yolov8n_soup\weights\soup_a0.50.pt",
}


def evaluar(modelo_path, nombre_run):
    modelo = YOLO(modelo_path)
    metricas = modelo.val(
        data=DATA_PROPIO,
        split="val",
        imgsz=IMGSZ,
        iou=IOU_NMS,
        device=0,
        workers=0,
        project=rf"{RAIZ}\yolo_runs",
        name=nombre_run,
        verbose=False,
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"P": p, "R": r, "mAP50": metricas.box.map50, "F1": f1}


if __name__ == "__main__":
    resultados = {}
    for nombre, ruta in MODELOS.items():
        print(f"\n{'=' * 50}\n{nombre}\n{'=' * 50}")
        r = evaluar(ruta, f"whitefly_propio_CORREGIDO_{nombre.replace('=', '').replace('.', '')}")
        resultados[nombre] = r
        print(f"  P={r['P']:.4f} R={r['R']:.4f} mAP50={r['mAP50']:.4f} F1={r['F1']:.4f}")

    print("\n" + "=" * 80)
    print("TEST PROPIO CORREGIDO (Etapa 15) -- TODOS los modelos, iou=0.45")
    print("=" * 80)
    print(f"{'Modelo':>14} | {'P':>8} | {'R':>8} | {'mAP50':>8} | {'F1':>8}")
    for nombre, r in resultados.items():
        print(f"{nombre:>14} | {r['P']:>8.4f} | {r['R']:>8.4f} | {r['mAP50']:>8.4f} | {r['F1']:>8.4f}")
    print("=" * 80)
