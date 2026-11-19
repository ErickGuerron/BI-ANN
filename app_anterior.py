"""Front de la red neuronal (Streamlit) para los 8 datasets de cartera v4.

Secciones: Entrenamiento (en vivo, con el lienzo de neuronas), Resultados y Prueba.
Solo la tasa de aprendizaje se edita. Las épocas se detienen por parada temprana sobre validación.
Ejecutar:  streamlit run app.py
"""
import json
import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import balanced_accuracy_score, classification_report

import ann_core
from ann_core import (
    DATASETS,
    PACIENCIA,
    TOL_VALIDACION,
    MAX_EPOCAS,
    ConfigRed,
    cargar_dataset,
    crear_modelo,
    preparar_train,
    receta_original,
    tamano_lote_sugerido,
)
from visual_red import COLOR_NAVY, html_lienzo

CARPETA = ann_core.CARPETA_RESULTADOS.parent / "resultados_convergencia"
NOMBRES = {
    "ds1_ingreso_solicitudes": "ds1 ingreso de solicitudes",
    "ds2_ciclo_solicitud": "ds2 ciclo de solicitud",
    "ds3_desembolsos": "ds3 desembolsos",
    "ds4_cierre_cartera_mensual": "ds4 cierre de cartera",
    "ds5_pagos_multimes": "ds5 pagos multimes",
    "ds6_gasto_provisiones": "ds6 gasto y provisiones",
    "ds7_historico_garantias": "ds7 histórico de garantías",
    "ds8_notificaciones_cobranza": "ds8 notificaciones de cobranza",
}
TASA_RECETA = 1e-3
SEMILLA = 42

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Instrument Sans', system-ui, sans-serif; color: #15233A; }
.stApp { background: #E7ECF1; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1440px; }
.disp { font-family: 'Newsreader', Georgia, serif; font-weight: 400; }
.mut { color: #4A5B70; }
.num { font-variant-numeric: tabular-nums; }
.navbar { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #B9C4D0; padding:0 0 14px 0; margin-bottom:10px; }
.marca { display:flex; align-items:center; gap:14px; }
.marca .disp { font-size:22px; line-height:1; }
.tarjeta { background:#F4F6F9; border:1px solid #B9C4D0; border-radius:2px; padding:14px 16px; }
.tarjeta.oscura { background:#15233A; border-color:#15233A; color:#F4F6F9; }
.tarjeta.oscura .mut { color:#C9D2DC; }
.tarjeta .valor { font-family:'Newsreader', Georgia, serif; font-size:30px; line-height:1.1; font-variant-numeric: tabular-nums; }
.tarjeta .valor.grande { font-size:56px; }
.tarjeta .nota { font-size:12px; color:#4A5B70; margin-top:4px; }
.tarjeta.oscura .nota { color:#C9D2DC; }
.lateral { background:#DDE4EC; border:1px solid #B9C4D0; border-radius:2px; padding:14px 16px; }
.fila { display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid #C9D2DC; font-size:14px; }
.fila:last-child { border-bottom:none; }
.barra { height:8px; background:#C9D2DC; position:relative; }
.barra > i { position:absolute; left:0; top:0; bottom:0; }
.chip { display:inline-block; font-size:12px; padding:4px 9px; border:1px solid #8C9BAD; border-radius:2px; background:#F4F6F9; }
table.tabla { border-collapse:separate; border-spacing:4px; font-size:14px; }
table.tabla th { font-weight:500; color:#4A5B70; font-size:12px; padding:4px 6px; text-align:center; }
table.tabla td { text-align:center; padding:10px 6px; border-radius:2px; font-variant-numeric: tabular-nums; }
table.tabla td.fila-h { text-align:right; color:#4A5B70; font-size:13px; padding-right:10px; background:none; }
div[data-testid="stTabs"] button[role="tab"] { font-size:15px; color:#4A5B70; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#15233A; font-weight:600; }
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] { background:#15233A; }
div[data-baseweb="select"] > div { background:#F4F6F9 !important; border:1px solid #8C9BAD !important; border-radius:2px !important; }
div[data-baseweb="select"] > div * { color:#15233A !important; }
div[data-testid="stNumberInput"] input { background:#F4F6F9 !important; color:#15233A !important; }
div[data-testid="stNumberInput"] div[data-baseweb="input"] { background:#F4F6F9 !important; border:1px solid #8C9BAD !important; border-radius:2px !important; }
div[data-testid="stNumberInput"] button { background:#F4F6F9 !important; color:#15233A !important; }
button[data-testid="stBaseButton-primary"] { background:#15233A !important; border:1px solid #15233A !important; color:#F4F6F9 !important; border-radius:2px !important; }
button[data-testid="stBaseButton-primary"]:hover { background:#4A5B70 !important; border-color:#4A5B70 !important; }
button[data-testid="stBaseButton-secondary"] { background:#F4F6F9 !important; border:1px solid #8C9BAD !important; color:#15233A !important; border-radius:2px !important; }
button[data-testid="stBaseButton-secondary"]:hover { border-color:#15233A !important; }
</style>
"""

LOGO = """<svg width="40" height="40" viewBox="0 0 36 36" aria-hidden="true">
<path d="M11 10 L16 6 M11 10 L16 18 M11 26 L16 18 M11 26 L16 30 M21 6 L25 16 M21 18 L25 18 M21 30 L25 20" stroke="#4A5B70" stroke-width="1.4" fill="none"/>
<circle cx="8" cy="10" r="3.6" fill="#B7791F"/><circle cx="8" cy="26" r="3.6" fill="#B7791F"/>
<circle cx="18" cy="6" r="3" fill="#15233A"/><circle cx="18" cy="18" r="3" fill="#15233A"/><circle cx="18" cy="30" r="3" fill="#15233A"/>
<circle cx="28" cy="18" r="4.2" fill="#B7791F"/></svg>"""


def tarjeta(etiqueta, valor, nota="", oscura=False, grande=False):
    clase = "tarjeta oscura" if oscura else "tarjeta"
    extra = " grande" if grande else ""
    return (f'<div class="{clase}"><div class="mut" style="font-size:12px">{etiqueta}</div>'
            f'<div class="valor{extra}">{valor}</div><div class="nota">{nota}</div></div>')


def fmt(x, dec=3):
    return f"{x:.{dec}f}".replace(".", ",")


@st.cache_data(show_spinner=False)
def cargar_ajuste(ds):
    ruta = CARPETA / f"ajuste_{ds}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def cargar_top3(ds):
    ruta = CARPETA / f"top3_{ds}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def seccion_top3(ds):
    top = cargar_top3(ds)
    if not top:
        return
    filas = []
    for c in top["configuraciones"]:
        capas = " → ".join(str(n) for n in c["capas_ocultas"])
        n_capas = c["numero_capas_ocultas"]
        arquitectura = f'{capas} neuronas · {n_capas} capa' + ("s" if n_capas > 1 else "")
        semillas = c["cv_por_semilla"]
        marca = '<span class="chip">Elegida</span>' if c["elegida"] else ""
        estilo = ' style="font-weight:600"' if c["elegida"] else ""
        filas.append(
            f'<tr{estilo}><td>{c["posicion"]}</td><td style="text-align:left">{arquitectura}</td>'
            f'<td>{fmt(c["alpha_l2"], 1)}</td><td>{c["tamano_lote"]}</td>'
            f'<td>{"Sí" if c["sobremuestreo"] else "No"}</td><td>{fmt(c["cv_promedio"])}</td>'
            f'<td>{fmt(min(semillas))} – {fmt(max(semillas))}</td><td>{c["epoca_mejor_cv"]}</td><td>{marca}</td></tr>'
        )
    st.markdown('<div class="disp" style="font-size:26px;margin:4px 0 6px">Tres mejores configuraciones</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<table class="tabla" style="width:100%"><thead><tr><th>#</th><th style="text-align:left">Arquitectura</th>'
        '<th>α</th><th>Lote</th><th>Sobremuestreo</th><th>CV balanceada</th><th>Rango por semilla</th>'
        '<th>Época</th><th></th></tr></thead><tbody>' + "".join(filas) + "</tbody></table>",
        unsafe_allow_html=True)
    st.markdown(f'<div class="mut" style="font-size:13px;margin-top:6px">{top["criterio"]} '
                'Solo la configuración elegida tiene métricas de validación y prueba.</div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cargar_pesos(ds):
    ruta = CARPETA / f"pesos_{ds}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


@st.cache_data(show_spinner=False)
def cargar_predicciones(ds):
    ruta = CARPETA / f"predicciones_prueba_{ds}.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data(show_spinner=False)
def cargar_prueba(ds):
    return pd.read_csv(ann_core.CARPETA_DATOS / ds / "X_test_nn.csv").to_numpy(dtype=np.float64)


@st.cache_data(show_spinner=False)
def nombres_entradas(ds):
    return list(pd.read_csv(ann_core.CARPETA_DATOS / ds / "X_train_nn.csv", nrows=0).columns)


@st.cache_data(show_spinner=False)
def clases_de(ds):
    etiquetas = pd.read_csv(ann_core.CARPETA_DATOS / ds / "y_train.csv")[["Y", "Etiqueta"]].drop_duplicates()
    return etiquetas.sort_values("Y")["Etiqueta"].tolist()


def propagar(pesos, x):
    """Pasada hacia adelante con los pesos guardados. Devuelve activaciones por capa (la última son probabilidades)."""
    activaciones = [np.asarray(x, dtype=np.float64)]
    a = activaciones[0]
    n = len(pesos["coefs"])
    for l, (W, b) in enumerate(zip(pesos["coefs"], pesos["intercepts"])):
        z = a @ np.asarray(W) + np.asarray(b)
        if l < n - 1:
            a = np.maximum(z, 0)
        else:
            z = z - z.max()
            e = np.exp(z)
            a = e / e.sum()
        activaciones.append(a)
    return activaciones


def svg_curvas(series, ancho=520, alto=220, etiquetas=None):
    """Gráfica de líneas en SVG. series: lista de (nombre, valores, color)."""
    if not series or not series[0][1]:
        return ""
    todos = [v for _, vals, _ in series for v in vals]
    lo, hi = min(todos), max(todos)
    rango = (hi - lo) or 1e-9
    n = max(len(vals) for _, vals, _ in series)
    x0, x1, y0, y1 = 44, ancho - 12, 14, alto - 28

    def px(i):
        return x0 + (x1 - x0) * (i / max(1, n - 1))

    def py(v):
        return y1 - (y1 - y0) * ((v - lo) / rango)

    partes = [f'<svg viewBox="0 0 {ancho} {alto}" width="100%" role="img">',
              f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#8C9BAD"/>',
              f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8C9BAD"/>']
    for k in range(3):
        v = lo + rango * k / 2
        partes.append(f'<text x="{x0 - 6}" y="{py(v) + 4}" font-size="11" fill="#4A5B70" text-anchor="end">{fmt(v, 3)}</text>')
    for nombre, vals, color in series:
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        partes.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i, v in enumerate(vals):
            partes.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3" fill="{color}"/>')
    if etiquetas:
        for i in range(n):
            partes.append(f'<text x="{px(i):.1f}" y="{alto - 8}" font-size="11" fill="#4A5B70" text-anchor="middle">{i + 1}</text>')
    partes.append("</svg>")
    return "".join(partes)


def config_de(ds, ajuste):
    """Configuración de partida: la del ajuste por convergencia, o la receta v4 si aún no hay ajuste."""
    if ajuste:
        c = ajuste["config_elegida"]
        return {"capas": tuple(c["capas_ocultas"]), "alpha": c["alpha"], "lote": c["tamano_lote"],
                "sobre": c["sobremuestreo"], "origen": "Ajuste por convergencia"}
    p = receta_original(ds)["parametros"]
    datos = cargar_dataset(ds)
    return {"capas": tuple(p["hidden"]), "alpha": p["alpha"], "lote": tamano_lote_sugerido(len(datos["train"]["y"])),
            "sobre": p["sobre"], "origen": "Receta v4 (sin ajuste todavía)"}


def seccion_entrenamiento(ds, ajuste):
    cfg0 = config_de(ds, ajuste)
    datos = cargar_dataset(ds)
    n_in = datos["train"]["X"].shape[1]
    clases = clases_de(ds)
    capas = [n_in, *cfg0["capas"], len(clases)]
    izq, centro, der = st.columns([1, 2.3, 1.25], gap="large")

    with izq:
        st.markdown('<div class="disp" style="font-size:26px">Configuración</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mut" style="font-size:13px;line-height:1.5;margin:6px 0 12px">'
                    f'{cfg0["origen"]}. Solo la tasa de aprendizaje se puede cambiar.</div>', unsafe_allow_html=True)
        celdas = "".join(
            f'<div class="tarjeta" style="flex:1;text-align:center;padding:8px 4px;{"border-color:#15233A" if 0 < i < len(capas) - 1 else ""}">'
            f'<div class="disp num" style="font-size:22px">{n}</div><div class="mut" style="font-size:11px">'
            f'{["entrada", *[f"oculta {k}" for k in range(1, len(capas) - 1)], "salida"][i]}</div></div>'
            for i, n in enumerate(capas))
        st.markdown(f'<div style="display:flex;gap:6px;align-items:center">{celdas}</div>', unsafe_allow_html=True)
        st.markdown(tarjeta("Épocas", "Hasta convergencia",
                            f"Parada temprana: {PACIENCIA} épocas sin mejora de validación"), unsafe_allow_html=True)
        st.markdown(tarjeta("α, regularización L2", fmt(cfg0["alpha"], 1), "Fija por receta"), unsafe_allow_html=True)
        tasa = st.number_input("Tasa de aprendizaje", min_value=0.00001, max_value=1.0,
                               value=TASA_RECETA, step=0.0001, format="%.5f", key=f"tasa_{ds}")
        st.markdown(f'<div class="fila"><span class="mut">Lote</span><span class="num">{cfg0["lote"]}</span></div>'
                    f'<div class="fila"><span class="mut">Sobremuestreo, solo en train</span><span>{"Sí" if cfg0["sobre"] else "No"}</span></div>',
                    unsafe_allow_html=True)
        entrenar = st.button("Entrenar con esta tasa", type="primary", use_container_width=True)
        if st.button("Restablecer tasa", use_container_width=True):
            st.session_state[f"tasa_{ds}"] = TASA_RECETA
            st.rerun()

    with centro:
        cabecera = st.empty()
        indicadores = st.empty()
        lienzo_slot = st.empty()
        estado = st.empty()

    with der:
        st.markdown('<div class="disp" style="font-size:24px">Validación por época</div>', unsafe_allow_html=True)
        grafica = st.empty()
        tabla = st.empty()

    if entrenar:
        cfg = ConfigRed(capas_ocultas=cfg0["capas"], alpha=cfg0["alpha"], tamano_lote=cfg0["lote"],
                        tasa_aprendizaje=float(tasa), sobremuestreo=cfg0["sobre"], semilla=SEMILLA)
        X, y = preparar_train(cfg, datos["train"]["X"], datos["train"]["y"])
        modelo = crear_modelo(cfg)
        clases_np = np.unique(datos["train"]["y"])
        historial, mejor, mejor_epoca, mejor_pesos, sin_mejora = [], -np.inf, 0, None, 0
        for epoca in range(1, MAX_EPOCAS + 1):
            modelo.partial_fit(X, y, classes=clases_np)
            score = float(balanced_accuracy_score(datos["val"]["y"], modelo.predict(datos["val"]["X"])))
            historial.append({"epoca": epoca, "perdida": float(modelo.loss_), "val": score})
            if score > mejor + TOL_VALIDACION:
                mejor, mejor_epoca, sin_mejora = score, epoca, 0
                mejor_pesos = ([w.copy() for w in modelo.coefs_], [b.copy() for b in modelo.intercepts_])
            else:
                sin_mejora += 1
            cabecera.markdown(f'<div style="display:flex;align-items:center;gap:8px;font-size:14px;font-weight:600">'
                              f'<span style="width:9px;height:9px;border-radius:50%;background:#15233A"></span>Entrenando</div>'
                              f'<div class="disp num" style="font-size:60px;line-height:1.02">Época {epoca}</div>', unsafe_allow_html=True)
            indicadores.markdown(
                '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">'
                + tarjeta("Pérdida, entrenamiento", fmt(historial[-1]["perdida"]))
                + tarjeta("Exactitud balanceada, validación", fmt(score))
                + tarjeta("Mejor validación hasta ahora", fmt(mejor), f"época {mejor_epoca}")
                + "</div>", unsafe_allow_html=True)
            lienzo_slot.empty()
            with lienzo_slot.container():
                st.iframe(html_lienzo(capas, modelo.coefs_, modo="entrenamiento", entradas=nombres_entradas(ds),
                                            clases=clases, alto=520), height=540)
            grafica.markdown(svg_curvas([("validación", [h["val"] for h in historial], "#15233A")]), unsafe_allow_html=True)
            if sin_mejora == PACIENCIA:
                break
            time.sleep(0.02)
        estado.markdown(f'<div class="mut" style="font-size:13px">Terminó en la época {len(historial)}; '
                        f'la mejor fue la {mejor_epoca}, con validación {fmt(mejor)}. Los pesos de esa época son los que se muestran.</div>',
                        unsafe_allow_html=True)
        st.session_state[f"ultimo_{ds}"] = {"historial": historial, "mejor_epoca": mejor_epoca}
        return

    # Vista inicial: pesos guardados del ajuste, o pesos de una red sin entrenar en esta sesión
    pesos = cargar_pesos(ds)
    if pesos:
        cabecera.markdown('<div class="disp num" style="font-size:36px;line-height:1.1;color:#15233A">Modelo guardado</div>', unsafe_allow_html=True)
        indicadores.markdown(
            '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">'
            + tarjeta("Exactitud balanceada, validación", fmt(ajuste["validacion"]["exactitud_balanceada"]) if ajuste else "—")
            + tarjeta("Épocas hasta la mejor validación", str(pesos["epoca_final"]))
            + tarjeta("Casos de entrenamiento", f'{len(datos["train"]["y"]):,}'.replace(",", " "), f"{len(clases)} clases · {n_in} variables")
            + "</div>", unsafe_allow_html=True)
        with lienzo_slot.container():
            st.iframe(html_lienzo(pesos["capas"], pesos["coefs"], modo="entrenamiento", entradas=pesos["entradas"],
                                        clases=clases, alto=520), height=540)
        estado.markdown('<div class="mut" style="font-size:13px">Pulsa "Entrenar con esta tasa" para entrenar la red '
                        'desde cero y ver cómo cambian las conexiones época a época.</div>', unsafe_allow_html=True)
    else:
        cabecera.markdown('<div class="disp" style="font-size:40px">Sin pesos guardados</div>', unsafe_allow_html=True)
        estado.markdown('<div class="mut" style="font-size:13px">Este dataset todavía no tiene un modelo ajustado. '
                        'Pulsa "Entrenar con esta tasa" para entrenarlo.</div>', unsafe_allow_html=True)
    ultimo = st.session_state.get(f"ultimo_{ds}")
    if ultimo:
        grafica.markdown(svg_curvas([("validación", [h["val"] for h in ultimo["historial"]], "#15233A")]), unsafe_allow_html=True)


@st.dialog("Curva de aprendizaje", width="large")
def modal_curva(ajuste):
    h = ajuste["historial"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Pérdida de entrenamiento**")
        st.markdown(svg_curvas([("pérdida", [x["perdida_train"] for x in h], "#15233A")]), unsafe_allow_html=True)
    with c2:
        st.markdown("**Exactitud balanceada, validación**")
        st.markdown(svg_curvas([("validación", [x["exactitud_balanceada_val"] for x in h], "#6B7F99")]), unsafe_allow_html=True)
    st.markdown(f'<div class="mut" style="font-size:13px">La parada es temprana sobre validación. La mejor época fue la {ajuste["epoca_final"]}.</div>',
                unsafe_allow_html=True)


@st.dialog("Búsqueda de hiperparámetros", width="large")
def modal_busqueda(ajuste):
    filas = ajuste["tabla_fase1"]
    cuadro = {}
    for f in filas:
        c = f["config"]
        cuadro[(c["alpha"], tuple(c["capas_ocultas"]), c["sobremuestreo"])] = f["cv"]
    alphas = sorted({k[0] for k in cuadro})
    arq = [(32,), (64, 32), (128, 64)]
    html = ['<table class="tabla"><thead><tr><th>α</th>']
    for sobre in (False, True):
        for a in arq:
            html.append(f'<th>{"con" if sobre else "sin"} sobremuestreo<br>{" → ".join(map(str, a))}</th>')
    html.append("</tr></thead><tbody>")
    for alpha in alphas:
        html.append(f'<tr><td class="fila-h">{fmt(alpha, 4)}</td>')
        for sobre in (False, True):
            for a in arq:
                v = cuadro.get((alpha, a, sobre))
                fondo = "#DDE4EC" if v is None or v < 0.49 else "#8C9BAD" if v < 0.52 else "#4A5B70" if v < 0.54 else "#15233A"
                color = "#15233A" if fondo in ("#DDE4EC", "#8C9BAD") else "#F4F6F9"
                html.append(f'<td style="background:{fondo};color:{color}">{fmt(v) if v is not None else "—"}</td>')
        html.append("</tr>")
    html.append("</tbody></table>")
    st.markdown("".join(html), unsafe_allow_html=True)
    st.markdown(f'<div class="mut" style="font-size:13px">Exactitud balanceada de validación cruzada sobre train, por α, arquitectura y sobremuestreo. '
                f'Elegido: {ajuste["config_elegida"]["capas_ocultas"]}, α {fmt(ajuste["config_elegida"]["alpha"], 1)}, '
                f'k = {ajuste.get("folds", 5)} folds.</div>', unsafe_allow_html=True)


def seccion_resultados(ds, ajuste):
    if not ajuste:
        st.info("Este dataset todavía no tiene ajuste por convergencia. Usa la pestaña Entrenamiento.")
        return
    seccion_top3(ds)
    pred = cargar_predicciones(ds)
    clases = ajuste["clases"]
    prueba = ajuste["prueba"]
    izq, centro = st.columns([1.25, 1])
    st.markdown(
        '<div style="display:grid;grid-template-columns:1.25fr 1fr 1fr 1fr;gap:12px">'
        + tarjeta("Exactitud balanceada, prueba", fmt(prueba["exactitud_balanceada"]), f'receta anterior {fmt(ajuste["receta_v4_prueba_bal"] or 0)}', oscura=True, grande=True)
        + tarjeta("Exactitud balanceada, validación", fmt(ajuste["validacion"]["exactitud_balanceada"]), "usada para parar")
        + tarjeta("Validación cruzada, 4 semillas", fmt(ajuste["cv_promedio"]), f'{fmt(min(ajuste["cv_por_semilla"]))} a {fmt(max(ajuste["cv_por_semilla"]))}')
        + tarjeta("AUC macro, prueba", fmt(prueba["auc_macro_ovr"]), f'exactitud {fmt(prueba["exactitud"])}')
        + "</div>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1.1, 1.1, 0.9], gap="large")
    with c1:
        st.markdown('<div class="disp" style="font-size:24px">Matriz de confusión</div>', unsafe_allow_html=True)
        cm = pd.crosstab(pred["clase_real"], pred["clase_predicha"]).reindex(index=clases, columns=clases, fill_value=0)
        filas_pct = cm.div(cm.sum(axis=1), axis=0)
        html = ['<table class="tabla"><thead><tr><th></th>' + "".join(f"<th>{c}</th>" for c in clases) + "</tr></thead><tbody>"]
        for r in clases:
            html.append(f"<tr><td class=\"fila-h\" style=\"text-align:left\">{r}</td>")
            for c in clases:
                p = filas_pct.loc[r, c]
                fondo = "#15233A" if p >= 0.5 else "#4A5B70" if p >= 0.3 else "#8C9BAD" if p >= 0.2 else "#DDE4EC"
                color = "#F4F6F9" if p >= 0.3 else "#15233A"
                html.append(f'<td style="background:{fondo};color:{color};font-weight:600">{cm.loc[r, c]:,}<br><span style="font-size:12px;font-weight:400">{p * 100:.0f} %</span></td>'.replace(",", " "))
            html.append("</tr>")
        html.append("</tbody></table>")
        st.markdown("".join(html), unsafe_allow_html=True)
        st.markdown('<div class="mut" style="font-size:12px">Filas: clase real. Columnas: clase predicha. La diagonal es el acierto.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="disp" style="font-size:24px">Métricas por clase</div>', unsafe_allow_html=True)
        rep = classification_report(pred["clase_real"], pred["clase_predicha"], labels=clases, output_dict=True, zero_division=0)
        filas = "".join(
            f'<tr><td style="text-align:left">{c}</td><td>{int(rep[c]["support"])}</td><td>{fmt(rep[c]["precision"])}</td>'
            f'<td style="font-weight:600">{fmt(rep[c]["recall"])}</td><td>{fmt(rep[c]["f1-score"])}</td></tr>' for c in clases)
        st.markdown('<table class="tabla" style="width:100%"><thead><tr><th style="text-align:left">Clase</th><th>Casos reales</th>'
                    '<th>Precisión</th><th>Recall</th><th>F1</th></tr></thead><tbody>' + filas + "</tbody></table>", unsafe_allow_html=True)
        st.markdown('<div class="mut" style="font-size:13px;margin-top:8px">Recall y F1 muestran el costo del desbalance; la precisión de la clase mayoritaria es alta por tamaño.</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("Curva de aprendizaje", use_container_width=True):
            modal_curva(ajuste)
        if b2.button("Búsqueda de hiperparámetros", use_container_width=True):
            modal_busqueda(ajuste)
    with c3:
        cfg = ajuste["config_elegida"]
        st.markdown('<div class="disp" style="font-size:24px">Configuración elegida</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="lateral">'
            f'<div class="fila"><span class="mut">Capas ocultas</span><span class="num">{" → ".join(map(str, cfg["capas_ocultas"]))}</span></div>'
            f'<div class="fila"><span class="mut">α, regularización L2</span><span class="num">{fmt(cfg["alpha"], 1)}</span></div>'
            f'<div class="fila"><span class="mut">Tasa de aprendizaje</span><span class="num">{fmt(cfg["tasa_aprendizaje"], 4)}</span></div>'
            f'<div class="fila"><span class="mut">Lote</span><span class="num">{cfg["tamano_lote"]}</span></div>'
            f'<div class="fila"><span class="mut">Sobremuestreo</span><span>{"Sí" if cfg["sobremuestreo"] else "No"}</span></div>'
            f'<div class="fila"><span class="mut">Épocas hasta la mejor validación</span><span class="num">{ajuste["epoca_final"]}</span></div>'
            "</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="disp" style="font-size:24px">Errores destacados</div>', unsafe_allow_html=True)
        errores = pred[~pred["acierto"]].head(3)
        filas = "".join(
            f'<div class="fila"><span class="num">{r.id}</span><span>{r.clase_real} → {r.clase_predicha}</span></div>'
            for r in errores.itertuples())
        st.markdown(f'<div class="lateral">{filas}</div>', unsafe_allow_html=True)
        st.download_button("Descargar predicciones (CSV)", pred.to_csv(index=False).encode("utf-8"),
                           file_name=f"predicciones_{ds}.csv", mime="text/csv", use_container_width=True)


def seccion_prueba(ds, ajuste):
    pesos = cargar_pesos(ds)
    pred = cargar_predicciones(ds)
    if not pesos or pred is None:
        st.info("Este dataset todavía no tiene un modelo ajustado para probar casos.")
        return
    X = cargar_prueba(ds)
    clases = pesos["clases"]
    muestra = pred.sample(12, random_state=7)
    izq, centro, der = st.columns([0.9, 2.2, 1.1], gap="large")
    with izq:
        st.markdown('<div class="disp" style="font-size:26px">Casos de prueba</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mut" style="font-size:13px;margin:6px 0 12px">Muestra de 12 de los {len(pred):,} casos del conjunto de prueba.</div>'.replace(",", " "),
                    unsafe_allow_html=True)
        opciones = {int(i): f"{pred.loc[i, 'id']}  ·  {'✓' if pred.loc[i, 'acierto'] else '✗'}" for i in muestra.index}
        indice = st.radio("Caso", list(opciones), format_func=lambda i: opciones[i], label_visibility="collapsed")
    fila = pred.loc[indice]
    acts = propagar(pesos, X[indice])
    probas = acts[-1]
    real, predicho = fila["clase_real"], fila["clase_predicha"]
    with centro:
        st.markdown(f'<div class="mut" style="font-size:13px">Caso seleccionado</div>'
                    f'<div class="disp num" style="font-size:56px;line-height:1">{fila["id"]}</div>', unsafe_allow_html=True)
        st.iframe(html_lienzo(pesos["capas"], pesos["coefs"], modo="prueba", entradas=pesos["entradas"],
                                    clases=clases, activaciones=acts, probas=probas, alto=520), height=540)
        st.markdown('<div class="mut" style="font-size:13px">Las activaciones y contribuciones son reales: se calculan con los pesos guardados para este caso.</div>',
                    unsafe_allow_html=True)
    with der:
        color = "#15233A" if fila["acierto"] else "#9A5B2E"
        estado = "Predicción correcta" if fila["acierto"] else "Predicción incorrecta"
        st.markdown(f'<div class="tarjeta" style="border-color:{color}"><div style="color:{color};font-size:14px;font-weight:600">{estado}</div>'
                    f'<div class="disp" style="font-size:24px;margin-top:8px;line-height:1.2">Predijo {predicho} con {fmt(probas[clases.index(predicho)])}</div>'
                    f'<div class="mut" style="font-size:14px;margin-top:8px">La clase real es {real}, con probabilidad {fmt(probas[clases.index(real)])}.</div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="mut" style="font-size:12px;margin:14px 0 8px">Probabilidad por clase</div>', unsafe_allow_html=True)
        barras = ""
        for k, c in enumerate(clases):
            marca = " · real" if c == real else ""
            barras += (f'<div style="margin-bottom:12px;font-size:14px"><div style="display:flex;justify-content:space-between;margin-bottom:6px">'
                       f'<span>{c}<span class="mut">{marca}</span></span><span class="num" style="font-weight:600">{fmt(probas[k])}</span></div>'
                       f'<div class="barra"><i style="width:{probas[k] * 100:.1f}%;background:#15233A"></i></div></div>')
        st.markdown(barras, unsafe_allow_html=True)
    with izq:
        st.markdown('<div class="mut" style="font-size:12px;margin-top:10px">Pulsa un caso para verlo en el lienzo.</div>', unsafe_allow_html=True)


def encabezado():
    st.markdown(f'<div class="navbar"><div class="marca">{LOGO}<div><div class="disp">Red neuronal</div>'
                f'<div class="mut" style="font-size:12px;margin-top:4px">Cartera de crédito · datasets v4</div></div></div>'
                f'<div><span class="chip">Entrenamiento local · CPU</span></div></div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Red neuronal · datasets v4", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    encabezado()
    ds = st.selectbox("Dataset", DATASETS, format_func=lambda d: NOMBRES[d], label_visibility="collapsed")
    ajuste = cargar_ajuste(ds)
    t1, t2, t3 = st.tabs(["Entrenamiento", "Resultados", "Prueba"])
    with t1:
        seccion_entrenamiento(ds, ajuste)
    with t2:
        seccion_resultados(ds, ajuste)
    with t3:
        seccion_prueba(ds, ajuste)


if __name__ == "__main__":
    main()
