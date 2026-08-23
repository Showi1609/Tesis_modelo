"""
Reconstruye SOLO las etiquetas YOLO del test (no reentrena nada) a partir de los XML ya
corregidos (Etapa 15), y refresca yolo_dataset_desglose/labels/test_propio con las cajas nuevas.
Reutiliza las funciones de yolo_entrenamiento.py tal cual (mismo split, misma semilla=42) para
que el conjunto de test siga siendo exactamente el mismo, solo con las etiquetas corregidas.
"""

import os
import sys

sys.path.insert(0, r"C:\Users\jchag\Documents\TESIS")
import yolo_entrenamiento as yt

RAIZ = r"C:\Users\jchag\Documents\TESIS"
DIR_DESGLOSE = rf"{RAIZ}\yolo_dataset_desglose"

if __name__ == "__main__":
    print("--- Recolectando muestras (con XML ya corregidos) ---")
    muestras_por_fuente = yt.recolectar_muestras()
    train, val, test = yt.dividir_train_val_test(
        muestras_por_fuente, yt.PROPORCION_VAL, yt.PROPORCION_TEST, yt.SEMILLA
    )
    test_propio = [m for m in test if m["grupo"] == "propio"]
    print(f"{len(test_propio)} imagenes de test propio (mismo split de siempre, semilla={yt.SEMILLA})")

    dir_lbl_desglose = os.path.join(DIR_DESGLOSE, "labels", "test_propio")
    total_cajas = 0
    for muestra in test_propio:
        nombre = muestra["nombre_salida"]  # ya trae el prefijo "propio_"
        ruta_lbl = os.path.join(dir_lbl_desglose, nombre + ".txt")

        # normalizar usando las dimensiones de la imagen ORIGINAL (mismo espacio de coordenadas
        # que los xmin/ymin/xmax/ymax leidos del XML) -- NO la copia redimensionada de
        # yolo_dataset_desglose. Las coordenadas normalizadas son invariantes a la escala.
        from PIL import Image
        with Image.open(muestra["ruta_imagen"]) as img:
            img_w, img_h = img.size

        lineas = []
        for clase, xmin, ymin, xmax, ymax in muestra["cajas"]:
            clase_idx = yt.CLASES.index(clase)
            lineas.append(yt.caja_a_yolo(clase_idx, xmin, ymin, xmax, ymax, img_w, img_h))

        with open(ruta_lbl, "w") as f:
            f.write("\n".join(lineas))
        total_cajas += len(lineas)
        print(f"  {nombre}: {len(muestra['cajas'])} cajas (antes de corregir habia menos)")

    print(f"\n=== Total cajas WF en test_propio (corregido): {total_cajas} ===")
