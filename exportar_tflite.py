from ultralytics import YOLO

MODELO_BASE = r"C:\Users\jchag\Documents\TESIS\yolo_runs\whitefly_yolov8n_combinado-6\weights\best.pt"
DATA_YAML = r"C:\Users\jchag\Documents\TESIS\yolo_dataset_combinado\data.yaml"
IMGSZ = 1280

# NOTA IMPORTANTE (probado 2026-07-22): export(int8=True) genera VARIAS variantes .tflite en la misma
# carpeta best_saved_model/ (float32, float16, dynamic_range_quant, integer_quant). Comparadas sobre el
# test set: la cuantizacion INT8 estatica (integer_quant, con calibracion) COLAPSA el modelo (F1 0.73 -> 0.25),
# probablemente por pocas imagenes de calibracion (51, se recomiendan 300+) y porque la mosca blanca es un
# objeto muy pequeno, sensible a cuantizar tambien las activaciones. La variante buena es
# "best_dynamic_range_quant.tflite" (solo pesos en INT8, sin calibracion): F1=0.727, practicamente igual
# al modelo original, y ~3.3x mas liviana que FP32. Esa es la que se debe usar en la app Android.
if __name__ == "__main__":
    modelo = YOLO(MODELO_BASE)

    # int8=True dispara el pipeline via TensorFlow/onnx2tf (compatible con Windows) y de paso genera
    # las variantes anteriores; luego evaluamos directamente la dynamic-range, no esta ruta int8 "cruda"
    ruta_tflite = modelo.export(format="tflite", int8=True, data=DATA_YAML, imgsz=IMGSZ)
    print(f"\nModelo TFLite generado en: {ruta_tflite}")
    ruta_recomendada = ruta_tflite.replace("best_integer_quant.tflite", "best_dynamic_range_quant.tflite")

    print("\n--- Re-evaluando la variante recomendada (dynamic range quant) sobre el test set ---")
    modelo_tflite = YOLO(ruta_recomendada)
    metricas = modelo_tflite.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMGSZ,
        project=r"C:\Users\jchag\Documents\TESIS\yolo_runs",
        name="whitefly_yolov8n_tflite_int8_test",
    )

    print("\n" + "=" * 50)
    print("RESULTADOS DEL MODELO TFLITE INT8 SOBRE TEST")
    print("=" * 50)
    print(f"Precision (mp):     {metricas.box.mp:.4f}")
    print(f"Recall (mr):        {metricas.box.mr:.4f}")
    print(f"mAP50:              {metricas.box.map50:.4f}")
    print(f"mAP50-95:           {metricas.box.map:.4f}")
    p, r = metricas.box.mp, metricas.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"F1-Score:           {f1:.4f}")
    print("=" * 50)
