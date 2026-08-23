import tensorflow as tf

RUTA = r"C:\Users\jchag\Documents\TESIS\yolo_runs\whitefly_yolov8n_combinado-6\weights\best_saved_model\best_dynamic_range_quant.tflite"

interprete = tf.lite.Interpreter(model_path=RUTA)
interprete.allocate_tensors()

print("=== INPUT DETAILS ===")
for d in interprete.get_input_details():
    print(d)

print("\n=== OUTPUT DETAILS ===")
for d in interprete.get_output_details():
    print(d)
