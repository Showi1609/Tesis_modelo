# -*- coding: utf-8 -*-
"""Reevaluacion definitiva de Candidato A sobre el dominio propio.
Una sola pasada del pipeline por imagen; el emparejamiento codicioso se
recalcula desde cero en cada umbral (0,10 y 0,45), no se filtra."""
import sys, os, glob, json
sys.path.insert(0, "/mnt/user-data/uploads/TESIS")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cand_a", "/mnt/user-data/uploads/TESIS/candidato_a_iou045.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

BASE = "/mnt/user-data/uploads/TESIS/Propio/Tesis.voc"
DIR_IMG = os.path.join(BASE, "imagess")
DIR_XML = os.path.join(BASE, "annotations")
SAL = "/tmp/claude-0/-home-claude/f72b1eb4-7e7d-55a6-ba58-053c208b8814/scratchpad/eval"

pre = m.PreprocesamientoMIPE(erosion_borde=0, margen_trampa=0.03, debug=False, dir_salida=SAL)
det = m.DetectorUnificado(debug=False, dir_salida=SAL)
ev10 = m.EvaluadorDataset(umbral_iou=0.10)
ev45 = m.EvaluadorDataset(umbral_iou=0.45)

imgs = sorted(glob.glob(os.path.join(DIR_IMG, "*.jpg")))
print(f"imagenes: {len(imgs)}", flush=True)

filas, descartadas = [], []
for ruta in imgs:
    nb = os.path.splitext(os.path.basename(ruta))[0]
    rxml = os.path.join(DIR_XML, f"{nb}.xml")
    if not os.path.exists(rxml):
        descartadas.append((nb, "sin XML")); continue
    gt_orig = ev10.parsear_xml(rxml)
    prep, escala, matriz, msg = pre.ejecutar_pipeline(ruta)
    if prep is None:
        descartadas.append((nb, msg)); print(f"  [{nb}] descartada: {msg}", flush=True); continue
    gt = ev10.transformar_cajas_gt(gt_orig, escala, matriz)
    _, conteo, pred = det.detectar_y_dibujar(prep.copy())
    t10, f10, n10 = ev10.evaluar_imagen(pred, gt)
    t45, f45, n45 = ev45.evaluar_imagen(pred, gt)
    filas.append({"nombre": nb, "numero_base": int(nb.split("_")[0]),
                  "n_real": len(gt_orig), "n_pred": conteo,
                  "tp_0.1": t10, "fp_0.1": f10, "fn_0.1": n10,
                  "tp_0.45": t45, "fp_0.45": f45, "fn_0.45": n45})
    print(f"  [{nb}] real {len(gt_orig):>4} pred {conteo:>4} | 0.10 TP{t10:>4} FP{f10:>4} FN{n10:>4} | 0.45 TP{t45:>4} FP{f45:>4} FN{n45:>4}", flush=True)

def glob_m(k):
    TP = sum(r[f"tp_{k}"] for r in filas); FP = sum(r[f"fp_{k}"] for r in filas); FN = sum(r[f"fn_{k}"] for r in filas)
    P = TP/(TP+FP) if TP+FP else 0.0; R = TP/(TP+FN) if TP+FN else 0.0
    F = 2*P*R/(P+R) if P+R else 0.0
    return {"TP":TP,"FP":FP,"FN":FN,"precision":P,"recall":R,"f1":F}

out = {"n_imagenes_evaluadas": len(filas), "descartadas": descartadas,
       "global_0.1": glob_m("0.1"), "global_0.45": glob_m("0.45"), "por_imagen": filas}
with open(os.path.join(SAL, "candidato_a_propio_iou045.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nGLOBAL 0.10:", json.dumps(out["global_0.1"], indent=None))
print("GLOBAL 0.45:", json.dumps(out["global_0.45"], indent=None))
