"""
Weight averaging ("model soup") entre combinado-6 (best.pt) y finetune_propio_v2 (best.pt).

Que hace y por que puede funcionar aqui (y no en general):
- finetune_propio_v2 NO es un modelo independiente: es literalmente combinado-6 + 40 epocas
  adicionales de gradiente sobre las 168 imagenes propias (Etapa 7 de la bitacora). Ambos
  checkpoints comparten el mismo punto de partida en el espacio de pesos, solo que finetune_v2
  camino mas lejos en la direccion "propio" (y por eso olvido parte de "publico").
- Interpolar linealmente los pesos de dos checkpoints CERCANOS en la trayectoria de optimizacion
  (mismo arranque, misma arquitectura) tipicamente da un modelo que combina lo aprendido por
  ambos, en vez de romperse (a diferencia de promediar dos redes entrenadas por separado desde
  cero, donde las neuronas no estan alineadas y el promedio suele ser basura). Es la misma idea
  detras de "model soups" (Wortsman et al. 2022) y de SWA.
- No es "equilibrar el dataset" (eso ya paso en el entrenamiento via sobremuestreo/copy-paste).
  Aqui se mezclan PARAMETROS ya entrenados, buscando un punto intermedio entre "generalista"
  (combinado-6: fuerte en publico, mas debil en propio) y "sesgado a propio" (finetune_v2:
  parejo pero mas debil en publico) que reduzca el domain gap sin perder tanto en publico.

Para cada alpha en ALPHAS (peso de finetune_v2; 1-alpha es el peso de combinado-6):
  merged[k] = (1 - alpha) * combinado6[k] + alpha * finetune_v2[k]   (solo tensores float;
  buffers enteros como num_batches_tracked se toman de combinado-6 sin promediar)

Guarda cada mezcla como un .pt nuevo y la evalua sobre:
  - test agregado (53 img, yolo_dataset_combinado/data.yaml, split=test)
  - test_publico y test_propio (yolo_dataset_desglose, subsets ya construidos)

alpha=0.0 y alpha=1.0 deben reproducir (aprox) las cifras ya conocidas de combinado-6 y
finetune_v2 respectivamente -> sirve de sanity check de que el merge esta bien hecho.
"""

import copy
import os

import torch
from ultralytics import YOLO

RAIZ = r"C:\Users\jchag\Documents\TESIS"
MODELO_A = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt"       # generalista
MODELO_B = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_finetune_propio_v2\weights\best.pt"  # sesgado a propio
DATA_AGREGADO = rf"{RAIZ}\yolo_dataset_combinado\data.yaml"
DATA_PUBLICO = rf"{RAIZ}\yolo_dataset_desglose\data_test_publico.yaml"
DATA_PROPIO = rf"{RAIZ}\yolo_dataset_desglose\data_test_propio.yaml"
DIR_SOUP = rf"{RAIZ}\yolo_runs\whitefly_yolov8n_soup\weights"
IMGSZ = 1280

ALPHAS = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]


def cargar_state_dict_congelado(ruta):
    modelo = YOLO(ruta)
    return {k: v.clone() for k, v in modelo.model.state_dict().items()}, modelo


def mezclar(sd_a, sd_b, alpha):
    merged = {}
    for k, va in sd_a.items():
        vb = sd_b[k]
        if torch.is_floating_point(va):
            merged[k] = ((1 - alpha) * va.float() + alpha * vb.float()).to(va.dtype)
        else:
            merged[k] = va.clone()
    return merged


def evaluar(ruta_pt, data_yaml, nombre_run):
    modelo = YOLO(ruta_pt)
    metricas = modelo.val(
        data=data_yaml,
        split="test" if data_yaml == DATA_AGREGADO else "val",
        imgsz=IMGSZ,
        device=0,
        workers=0,
        project=rf"{RAIZ}\yolo_runs",
        name=nombre_run,
        verbose=False,
    )
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"P": p, "R": r, "mAP50": metricas.box.map50, "mAP50-95": metricas.box.map, "F1": f1}


if __name__ == "__main__":
    os.makedirs(DIR_SOUP, exist_ok=True)

    print("Cargando checkpoints base...")
    sd_a, modelo_base = cargar_state_dict_congelado(MODELO_A)
    sd_b, _ = cargar_state_dict_congelado(MODELO_B)

    filas = []
    for alpha in ALPHAS:
        print(f"\n{'=' * 60}\nMezclando alpha={alpha} (0=combinado-6 puro, 1=finetune_v2 puro)\n{'=' * 60}")
        merged = mezclar(sd_a, sd_b, alpha)

        # Cargar la mezcla en una copia fresca del modelo base y guardarla como checkpoint valido
        modelo_soup = YOLO(MODELO_A)
        modelo_soup.model.load_state_dict(merged)
        ruta_pt = os.path.join(DIR_SOUP, f"soup_a{alpha:.2f}.pt")
        modelo_soup.save(ruta_pt)
        del modelo_soup

        tag = f"a{alpha:.2f}".replace(".", "")
        agregado = evaluar(ruta_pt, DATA_AGREGADO, f"whitefly_soup_{tag}_agregado")
        publico = evaluar(ruta_pt, DATA_PUBLICO, f"whitefly_soup_{tag}_publico")
        propio = evaluar(ruta_pt, DATA_PROPIO, f"whitefly_soup_{tag}_propio")
        gap = publico["F1"] - propio["F1"]

        filas.append((alpha, agregado, publico, propio, gap))
        print(f"  agregado F1={agregado['F1']:.4f}  publico F1={publico['F1']:.4f}  "
              f"propio F1={propio['F1']:.4f}  gap={gap:.4f}")

    print("\n" + "=" * 100)
    print(f"{'alpha':>6} | {'F1 agregado':>12} | {'F1 publico':>11} | {'F1 propio':>10} | {'gap (pub-pro)':>13}")
    print("-" * 100)
    for alpha, agregado, publico, propio, gap in filas:
        print(f"{alpha:>6.2f} | {agregado['F1']:>12.4f} | {publico['F1']:>11.4f} | "
              f"{propio['F1']:>10.4f} | {gap:>13.4f}")
    print("=" * 100)
    print(f"\nCheckpoints guardados en: {DIR_SOUP}")
