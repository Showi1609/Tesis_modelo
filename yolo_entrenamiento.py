import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Fuentes de datos (VOC: imágenes + anotaciones XML). "grupo" se usa para poder
# entrenar cada dataset por separado (ver MODO_DATASET) además de combinado.
FUENTES = [
    {
        "nombre": "md121",
        "grupo": "publico",
        "dir_imagenes": r"C:\Users\jchag\Documents\TESIS\md121\images",
        "dir_anotaciones": r"C:\Users\jchag\Documents\TESIS\md121\annotations",
    },
    {
        "nombre": "propio",
        "grupo": "propio",
        "dir_imagenes": r"C:\Users\jchag\Documents\TESIS\Propio\Tesis.voc\imagess",
        "dir_anotaciones": r"C:\Users\jchag\Documents\TESIS\Propio\Tesis.voc\annotations",
    },
]

# Qué fuente(s) usar en esta corrida: "combinado" (md121 + propio), "publico" (solo md121) o "propio" (solo Propio)
MODO_DATASET = "combinado"
FUENTES_ACTIVAS = [f for f in FUENTES if MODO_DATASET == "combinado" or f["grupo"] == MODO_DATASET]

# Copy-paste augmentation: imágenes sintéticas (fondo propio real + moscas recortadas del dataset público).
# Se agregan SOLO a train (nunca a val/test: no son fotos reales, contaminarían la evaluación).
INCLUIR_COPY_PASTE = True
DIR_COPY_PASTE_IMAGENES = r"C:\Users\jchag\Documents\TESIS\Propio_copypaste\images"
DIR_COPY_PASTE_ANOTACIONES = r"C:\Users\jchag\Documents\TESIS\Propio_copypaste\annotations"

CLASES = ["WF"]  # Solo mosca blanca, igual que Test_all_XML y Test_metricas_A
# Cada modo escribe en su propia carpeta para no pisar los datasets/resultados de los otros modos
DIR_DATASET_YOLO = rf"C:\Users\jchag\Documents\TESIS\yolo_dataset_{MODO_DATASET}"
# Imágenes originales hasta 5184x3456px agotan la RAM al decodificarlas (equipo con poca memoria libre);
# se redimensionan al construir el dataset ya que de todas formas se entrena a IMGSZ=1280.
MAX_DIM_DATASET = 1920
# Split 70/15/15 (train/val/test), tal como se documenta en la sección 1.5 del anteproyecto
PROPORCION_VAL = 0.15
PROPORCION_TEST = 0.15
SEMILLA = 42

# Sobremuestreo de "propio" SOLO en train, para compensar el desbalance de dominio (82% público / 18% propio).
# El factor se calcula automáticamente para que el train quede ~50/50 entre grupos; val/test nunca se tocan.
SOBREMUESTREAR_PROPIO = True  # el factor se recalcula solo; con copy-paste ya casi balanceado, debería salir ~x1

# Controla qué fases se ejecutan al correr el script
CONVERTIR_DATASET = True
ENTRENAR = True

# Parámetros de entrenamiento (Candidato B: YOLOv8-nano, según la matriz de decisión del anteproyecto)
MODELO_BASE = "yolov8n.pt"
EPOCAS = 100
IMGSZ = 1280  # Las moscas ocupan pocos píxeles en imágenes grandes (hasta 5184px); 640 las volvería invisibles
BATCH = 4  # Ajustado para 6GB de VRAM a imgsz=1280 (batch=8 provoca CUDA OOM)
PACIENCIA = 20


def parsear_voc(ruta_xml, clases_validas):
    """Extrae cajas [clase, xmin, ymin, xmax, ymax] del XML VOC, filtrando por clases_validas."""
    cajas = []
    try:
        root = ET.parse(ruta_xml).getroot()
        for obj in root.findall("object"):
            nombre_clase = obj.find("name").text
            if nombre_clase not in clases_validas:
                continue
            bndbox = obj.find("bndbox")
            cajas.append([
                nombre_clase,
                float(bndbox.find("xmin").text),
                float(bndbox.find("ymin").text),
                float(bndbox.find("xmax").text),
                float(bndbox.find("ymax").text),
            ])
    except Exception as e:
        print(f"Error leyendo {ruta_xml}: {e}")
    return cajas


def caja_a_yolo(clase_idx, xmin, ymin, xmax, ymax, img_w, img_h):
    x_centro = ((xmin + xmax) / 2.0) / img_w
    y_centro = ((ymin + ymax) / 2.0) / img_h
    ancho = (xmax - xmin) / img_w
    alto = (ymax - ymin) / img_h
    return f"{clase_idx} {x_centro:.6f} {y_centro:.6f} {ancho:.6f} {alto:.6f}"


def recolectar_muestras():
    """Empareja imágenes con su XML por fuente y devuelve lista de dicts."""
    muestras_por_fuente = {}
    for fuente in FUENTES_ACTIVAS:
        muestras = []
        rutas_xml = glob.glob(os.path.join(fuente["dir_anotaciones"], "*.xml"))
        for ruta_xml in rutas_xml:
            nombre_base = os.path.splitext(os.path.basename(ruta_xml))[0]
            ruta_img = os.path.join(fuente["dir_imagenes"], nombre_base + ".jpg")
            if not os.path.exists(ruta_img):
                continue
            cajas = parsear_voc(ruta_xml, CLASES)
            muestras.append({
                "nombre_salida": f"{fuente['nombre']}_{nombre_base}",
                "ruta_imagen": ruta_img,
                "cajas": cajas,
                "grupo": fuente["grupo"],
            })
        muestras_por_fuente[fuente["nombre"]] = muestras
        print(f"[{fuente['nombre']}] {len(muestras)} imágenes con anotación emparejada.")
    return muestras_por_fuente


def cargar_muestras_copy_paste():
    """Carga las imágenes sintéticas (copy-paste) generadas por copy_paste_augmentation.py.
    Se devuelven como muestras 'propio' listas para agregarse directo a train (nunca a val/test)."""
    muestras = []
    rutas_xml = glob.glob(os.path.join(DIR_COPY_PASTE_ANOTACIONES, "*.xml"))
    for ruta_xml in rutas_xml:
        nombre_base = os.path.splitext(os.path.basename(ruta_xml))[0]
        ruta_img = os.path.join(DIR_COPY_PASTE_IMAGENES, nombre_base + ".jpg")
        if not os.path.exists(ruta_img):
            continue
        cajas = parsear_voc(ruta_xml, CLASES)
        muestras.append({
            "nombre_salida": f"cp_{nombre_base}",
            "ruta_imagen": ruta_img,
            "cajas": cajas,
            "grupo": "propio",
        })
    return muestras


def dividir_train_val_test(muestras_por_fuente, proporcion_val, proporcion_test, semilla):
    """Divide train/val/test (70/15/15) por separado en cada fuente para mantener proporciones (estratificado por origen)."""
    train, val, test = [], [], []
    rng = random.Random(semilla)
    for nombre_fuente, muestras in muestras_por_fuente.items():
        copia = muestras[:]
        rng.shuffle(copia)
        n = len(copia)
        corte_train = int(n * (1 - proporcion_val - proporcion_test))
        corte_val = corte_train + int(n * proporcion_val)
        train.extend(copia[:corte_train])
        val.extend(copia[corte_train:corte_val])
        test.extend(copia[corte_val:])
    return train, val, test


def sobremuestrear_propio(train):
    """Duplica las muestras del grupo 'propio' SOLO dentro de train, para acercar su
    representación a la del resto (nunca se aplica a val/test: inflaría la métrica)."""
    propio = [m for m in train if m["grupo"] == "propio"]
    resto = [m for m in train if m["grupo"] != "propio"]
    if not propio or not resto:
        return train, 1

    factor = max(1, round(len(resto) / len(propio)))
    if factor <= 1:
        return train, 1

    propio_sobremuestreado = []
    for copia_n in range(factor):
        for m in propio:
            copia = dict(m)
            copia["nombre_salida"] = f"{m['nombre_salida']}_rep{copia_n}"
            propio_sobremuestreado.append(copia)

    return resto + propio_sobremuestreado, factor


def construir_dataset_yolo(train, val, test, dir_salida):
    for sub in ("images", "labels"):
        ruta_sub = os.path.join(dir_salida, sub)
        if os.path.isdir(ruta_sub):
            shutil.rmtree(ruta_sub)

    for split, muestras in [("train", train), ("val", val), ("test", test)]:
        os.makedirs(os.path.join(dir_salida, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dir_salida, "labels", split), exist_ok=True)

        for muestra in muestras:
            nombre = muestra["nombre_salida"]
            ruta_img_destino = os.path.join(dir_salida, "images", split, nombre + ".jpg")
            ruta_label_destino = os.path.join(dir_salida, "labels", split, nombre + ".txt")

            with Image.open(muestra["ruta_imagen"]) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size
                if max(img_w, img_h) > MAX_DIM_DATASET:
                    escala = MAX_DIM_DATASET / max(img_w, img_h)
                    img = img.resize((int(img_w * escala), int(img_h * escala)), Image.LANCZOS)
                img.save(ruta_img_destino, "JPEG", quality=95)

            # Las coordenadas normalizadas (fracción del ancho/alto) son invariantes a la escala,
            # así que se calculan con las dimensiones originales sin importar el redimensionado anterior.
            lineas = []
            for clase, xmin, ymin, xmax, ymax in muestra["cajas"]:
                clase_idx = CLASES.index(clase)
                lineas.append(caja_a_yolo(clase_idx, xmin, ymin, xmax, ymax, img_w, img_h))

            with open(ruta_label_destino, "w") as f:
                f.write("\n".join(lineas))

    ruta_yaml = os.path.join(dir_salida, "data.yaml")
    with open(ruta_yaml, "w") as f:
        f.write(f"path: {dir_salida}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n")
        for i, clase in enumerate(CLASES):
            f.write(f"  {i}: {clase}\n")

    return ruta_yaml


def contar_cajas(muestras):
    return sum(len(m["cajas"]) for m in muestras)


if __name__ == "__main__":
    ruta_data_yaml = os.path.join(DIR_DATASET_YOLO, "data.yaml")
    print(f"=== MODO_DATASET = {MODO_DATASET} ({', '.join(f['nombre'] for f in FUENTES_ACTIVAS)}) ===")

    if CONVERTIR_DATASET:
        print("--- Recolectando muestras (imagen + XML emparejados) ---")
        muestras_por_fuente = recolectar_muestras()

        print("\n--- Dividiendo train/val/test 70/15/15 (estratificado por fuente) ---")
        train, val, test = dividir_train_val_test(muestras_por_fuente, PROPORCION_VAL, PROPORCION_TEST, SEMILLA)
        print(f"Train: {len(train)} imágenes, {contar_cajas(train)} cajas WF")
        print(f"Val:   {len(val)} imágenes, {contar_cajas(val)} cajas WF")
        print(f"Test:  {len(test)} imágenes, {contar_cajas(test)} cajas WF")

        if INCLUIR_COPY_PASTE:
            muestras_cp = cargar_muestras_copy_paste()
            train.extend(muestras_cp)
            print(f"\n--- Copy-paste augmentation agregado a train ---")
            print(f"+{len(muestras_cp)} imágenes sintéticas, +{contar_cajas(muestras_cp)} cajas WF")
            print(f"Train total tras copy-paste: {len(train)} imágenes, {contar_cajas(train)} cajas WF")

        if SOBREMUESTREAR_PROPIO:
            n_propio_antes = len([m for m in train if m["grupo"] == "propio"])
            train, factor = sobremuestrear_propio(train)
            n_propio_despues = len([m for m in train if m["grupo"] == "propio"])
            print(f"\n--- Sobremuestreo de 'propio' en train (factor x{factor}) ---")
            print(f"Propio en train: {n_propio_antes} -> {n_propio_despues} (resto sin tocar)")
            print(f"Train total tras sobremuestreo: {len(train)} imágenes, {contar_cajas(train)} cajas WF")

        print(f"\n--- Construyendo dataset YOLO en: {DIR_DATASET_YOLO} ---")
        ruta_data_yaml = construir_dataset_yolo(train, val, test, DIR_DATASET_YOLO)
        print(f"data.yaml generado en: {ruta_data_yaml}")

    if ENTRENAR:
        from ultralytics import YOLO

        print(f"\n--- Entrenando {MODELO_BASE} sobre {ruta_data_yaml} ---")
        modelo = YOLO(MODELO_BASE)
        modelo.train(
            data=ruta_data_yaml,
            epochs=EPOCAS,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PACIENCIA,
            device=0,
            workers=0,  # dataloader workers en subprocesos rompen la carga de cublas64_12.dll en este equipo (Windows)
            degrees=15.0,  # rotaciones geométricas (Data Augmentation, sección 2.2.12 del anteproyecto)
            flipud=0.5,    # volteo vertical, además del horizontal (fliplr) que ya viene activado por defecto
            project=r"C:\Users\jchag\Documents\TESIS\yolo_runs",
            name=f"whitefly_yolov8n_{MODO_DATASET}",
        )
