"""Prueba de significancia estadistica (Wilcoxon signed-rank, pareado) entre Candidato A y
Candidato B (combinado-6_v2), sobre las 44 imagenes de TEST publico -- el unico subconjunto
donde ambos candidatos se evaluan sobre datos que B nunca vio en entrenamiento, con tamano de
muestra razonable (a diferencia del 5-fold de propio, que solo tiene n=5).

Mismo criterio para ambos: iou_match=0.1 (RD-03, el aprobado por el Comite).
Se excluyen imagenes sin moscas reales (F1 no esta definido sin positivos)."""

import glob
import os

from scipy.stats import wilcoxon
from ultralytics import YOLO

import candidato_a_publico_fix_exif_iou01 as cand_a

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DIR_TEST_YOLO_IMG = rf"{RAIZ}\yolo_dataset_desglose\images\test_publico"
DIR_TEST_YOLO_LBL = rf"{RAIZ}\yolo_dataset_desglose\labels\test_publico"
DIR_MD121_IMG = rf"{RAIZ}\md121\images"
DIR_MD121_ANN = rf"{RAIZ}\md121\annotations"
MODELO_B = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado6_v2\weights\best.pt"
IMGSZ = 1280
CONF = 0.25
IOU_NMS = 0.45
IOU_MATCH = 0.1


def f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def leer_gt_yolo(ruta_txt, img_w, img_h):
    cajas = []
    if not os.path.exists(ruta_txt):
        return cajas
    with open(ruta_txt) as fh:
        for linea in fh:
            partes = linea.split()
            if len(partes) < 5:
                continue
            _, cx, cy, w, h = map(float, partes[:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            cajas.append([x1, y1, x2, y2])
    return cajas


if __name__ == "__main__":
    rutas_test = sorted(glob.glob(os.path.join(DIR_TEST_YOLO_IMG, "*.jpg")))
    print(f"Imagenes en el test publico: {len(rutas_test)}")

    modelo_b = YOLO(MODELO_B)
    preprocesador = cand_a.PreprocesamientoMIPE(erosion_borde=0, margen_trampa=0.03, debug=False)
    detector_a = cand_a.DetectorUnificado(debug=False)
    evaluador_a = cand_a.EvaluadorDataset(umbral_iou=IOU_MATCH)

    filas = []
    for ruta_test in rutas_test:
        nombre_yolo = os.path.splitext(os.path.basename(ruta_test))[0]  # md121_XXXX
        nombre_orig = nombre_yolo.replace("md121_", "", 1)

        # --- Candidato B: YOLO sobre la imagen original, emparejamiento manual iou=0.1 ---
        ruta_lbl_yolo = os.path.join(DIR_TEST_YOLO_LBL, nombre_yolo + ".txt")
        resultado = modelo_b.predict(source=ruta_test, imgsz=IMGSZ, conf=CONF, iou=IOU_NMS,
                                      device=0, verbose=False)[0]
        img_h, img_w = resultado.orig_shape
        preds_b = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in resultado.boxes]
        gts_b = leer_gt_yolo(ruta_lbl_yolo, img_w, img_h)

        preds_ordenadas = sorted(preds_b, key=lambda p: -p[1])
        usados = [False] * len(gts_b)
        tp_b = 0
        for caja_pred, _ in preds_ordenadas:
            mejor_iou, mejor_j = 0, -1
            for j, gt in enumerate(gts_b):
                if usados[j]:
                    continue
                v = iou_xyxy(caja_pred, gt)
                if v > mejor_iou:
                    mejor_iou, mejor_j = v, j
            if mejor_iou >= IOU_MATCH:
                usados[mejor_j] = True
                tp_b += 1
        fp_b = len(preds_b) - tp_b
        fn_b = len(gts_b) - tp_b

        # --- Candidato A: sobre la imagen ORIGINAL (con su propio preprocesamiento) ---
        ruta_img_orig = os.path.join(DIR_MD121_IMG, nombre_orig + ".jpg")
        ruta_xml_orig = os.path.join(DIR_MD121_ANN, nombre_orig + ".xml")
        cajas_xml = evaluador_a.parsear_xml(ruta_xml_orig)
        img_prep, escala, matriz, msg = preprocesador.ejecutar_pipeline(ruta_img_orig)
        if img_prep is None:
            print(f"[{nombre_orig}] Candidato A: imagen descartada ({msg}) -- excluida del par")
            continue
        cajas_gt_a = evaluador_a.transformar_cajas_gt(cajas_xml, escala, matriz)
        _, _, cajas_pred_a = detector_a.detectar_y_dibujar(img_prep.copy())
        tp_a, fp_a, fn_a = evaluador_a.evaluar_imagen(cajas_pred_a, cajas_gt_a)

        n_real = len(gts_b)  # cantidad real de moscas en la imagen (para filtrar despues)
        f1_a = f1(tp_a, fp_a, fn_a)
        f1_b = f1(tp_b, fp_b, fn_b)
        filas.append((nombre_orig, n_real, tp_a, fp_a, fn_a, f1_a, tp_b, fp_b, fn_b, f1_b))
        print(f"[{nombre_orig}] real={n_real} | A: TP{tp_a} FP{fp_a} FN{fn_a} F1={f1_a:.4f} "
              f"| B: TP{tp_b} FP{fp_b} FN{fn_b} F1={f1_b:.4f}")

    # --- Filtra imagenes sin moscas reales (F1 no definido) ---
    pares = [f for f in filas if f[1] > 0]
    excluidas = len(filas) - len(pares)
    print(f"\nTotal evaluadas: {len(filas)}, excluidas (0 moscas reales): {excluidas}, "
          f"pares validos: {len(pares)}")

    f1s_a = [f[5] for f in pares]
    f1s_b = [f[9] for f in pares]
    diferencias = [b - a for a, b in zip(f1s_a, f1s_b)]

    print(f"\nF1 medio A: {sum(f1s_a)/len(f1s_a):.4f}")
    print(f"F1 medio B: {sum(f1s_b)/len(f1s_b):.4f}")
    print(f"Diferencia media (B-A): {sum(diferencias)/len(diferencias):.4f}")
    n_favor_b = sum(1 for d in diferencias if d > 0)
    n_favor_a = sum(1 for d in diferencias if d < 0)
    n_empate = sum(1 for d in diferencias if d == 0)
    print(f"Imagenes donde B > A: {n_favor_b}, A > B: {n_favor_a}, empate: {n_empate}")

    stat, p = wilcoxon(f1s_b, f1s_a, alternative="greater")
    print(f"\nWilcoxon signed-rank (H1: F1_B > F1_A), una cola: estadistico={stat:.2f}, p={p:.5f}")
    stat2, p2 = wilcoxon(f1s_b, f1s_a, alternative="two-sided")
    print(f"Wilcoxon signed-rank, dos colas: estadistico={stat2:.2f}, p={p2:.5f}")

    import json
    with open("wilcoxon_publico_resultados.json", "w") as fjson:
        json.dump({
            "n_total": len(filas), "n_excluidas": excluidas, "n_pares": len(pares),
            "f1_medio_a": sum(f1s_a) / len(f1s_a), "f1_medio_b": sum(f1s_b) / len(f1s_b),
            "n_favor_b": n_favor_b, "n_favor_a": n_favor_a, "n_empate": n_empate,
            "wilcoxon_una_cola_stat": float(stat), "wilcoxon_una_cola_p": float(p),
            "wilcoxon_dos_colas_stat": float(stat2), "wilcoxon_dos_colas_p": float(p2),
            "filas": filas,
        }, fjson, indent=2)
    print("\nGuardado: wilcoxon_publico_resultados.json")
