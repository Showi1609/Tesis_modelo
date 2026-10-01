# -*- coding: utf-8 -*-
"""Candidato A sobre las 44 imagenes de prueba del repositorio publico (las mismas
en que se evalua B). Una pasada del pipeline por imagen; emparejamiento recalculado
desde cero a IoU 0,10 y 0,45. Mismo modulo que test_wilcoxon_publico.py (fix EXIF)."""
import os, json, importlib.util
BASE = r"C:\Users\jchag\Documents\TESIS"
spec = importlib.util.spec_from_file_location("ca", f"{BASE}/candidato_a_publico_fix_exif_iou01.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ids = [r[0] for r in json.load(open(f"{BASE}/wilcoxon_publico_resultados.json"))["filas"]]
ref = {r[0]: r for r in json.load(open(f"{BASE}/wilcoxon_publico_resultados.json"))["filas"]}
pre = m.PreprocesamientoMIPE(erosion_borde=0, margen_trampa=0.03, debug=False)
det = m.DetectorUnificado(debug=False)
e10, e45 = m.EvaluadorDataset(umbral_iou=0.10), m.EvaluadorDataset(umbral_iou=0.45)
filas = []
for i in ids:
    gt0 = e10.parsear_xml(f"{BASE}/md121/annotations/{i}.xml")
    prep, esc, mat, msg = pre.ejecutar_pipeline(f"{BASE}/md121/images/{i}.jpg")
    if prep is None:
        print(f"[{i}] descartada: {msg}"); continue
    gt = e10.transformar_cajas_gt(gt0, esc, mat)
    _, n, pred = det.detectar_y_dibujar(prep.copy())
    a = e10.evaluar_imagen(pred, gt); b = e45.evaluar_imagen(pred, gt)
    r = ref[i]
    filas.append(dict(id=i, n_real=len(gt0), n_pred=n, tp10=a[0], fp10=a[1], fn10=a[2],
                      tp45=b[0], fp45=b[1], fn45=b[2], ref10=(r[2], r[3], r[4]), n_real_ref=r[1]))
def agg(k):
    T = sum(f[f"tp{k}"] for f in filas); P = sum(f[f"fp{k}"] for f in filas); N = sum(f[f"fn{k}"] for f in filas)
    return dict(TP=T, FP=P, FN=N, precision=T/(T+P), recall=T/(T+N), f1=2*T/(2*T+P+N))
out = dict(n=len(filas), g10=agg("10"), g45=agg("45"), filas=filas)
json.dump(out, open(os.path.join(BASE, "candidato_a_publico44.json"), "w"), indent=1)
dif = [f for f in filas if (f["tp10"], f["fp10"], f["fn10"]) != tuple(f["ref10"])]
print("imagenes:", len(filas), "| n_real distinto del ref:", [f["id"] for f in filas if f["n_real"] != f["n_real_ref"]])
print("filas a 0,10 distintas de tu corrida:", len(dif))
for f in dif[:8]: print("  ", f["id"], "nuevo", (f["tp10"], f["fp10"], f["fn10"]), "tuyo", tuple(f["ref10"]))
print("0,10:", {k: round(v, 4) if isinstance(v, float) else v for k, v in out["g10"].items()})
print("0,45:", {k: round(v, 4) if isinstance(v, float) else v for k, v in out["g45"].items()})
