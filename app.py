"""Front de clasificación para las 4 redes del top 4 (por AUC de prueba).

Cada dataset tiene su dashboard de resultados y un asistente de ingreso de datos en pasos (modal).
Los campos se validan contra el dataset (tipo, enteros, no negativos, rango observado y valores conocidos)
sin usar combobox, para poder probar casos nuevos. Al clasificar, el dato se convierte a la entrada de la red.
Ejecutar:  streamlit run app.py
"""
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "nucleo"))

import frontera  # noqa: E402
import inferencia as I  # noqa: E402

CARPETA = I.PROGRAMA / "resultados_convergencia"
TOP4 = [
    ("ds7_historico_garantias", "ds7 · histórico de garantías"),
    ("ds1_ingreso_solicitudes", "ds1 · ingreso de solicitudes"),
    ("ds6_gasto_provisiones", "ds6 · gasto y provisiones"),
    ("ds5_pagos_multimes", "ds5 · pagos multimes"),
]
ETIQUETAS = {
    "Anio_Desembolso": "Año de desembolso", "Valor_Entregado": "Valor entregado", "Numero_Cuotas": "Número de cuotas",
    "Tasa_Interes": "Tasa de interés", "Monto_por_Cuota": "Monto por cuota",
}
CAMPOS_POR_PASO = 6
COLOR_NAVY, COLOR_TINTA = "#15233A", "#4A5B70"

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
.tarjeta { background:#F4F6F9; border:1px solid #B9C4D0; border-radius:2px; padding:14px 16px; }
.tarjeta.oscura { background:#15233A; border-color:#15233A; color:#F4F6F9; }
.tarjeta.oscura .mut { color:#C9D2DC; }
.tarjeta .valor { font-family:'Newsreader', Georgia, serif; font-size:30px; line-height:1.1; font-variant-numeric: tabular-nums; }
.tarjeta .nota { font-size:12px; color:#4A5B70; margin-top:4px; }
.tarjeta.oscura .nota { color:#C9D2DC; }
.barra { height:8px; background:#C9D2DC; position:relative; }
.barra > i { position:absolute; left:0; top:0; bottom:0; }
.chip { display:inline-block; font-size:12px; padding:4px 9px; border:1px solid #8C9BAD; border-radius:2px; background:#F4F6F9; }
.aviso { font-size:13px; padding:8px 10px; border-left:3px solid #9A5B2E; background:#F4F6F9; margin:4px 0; }
.error { font-size:13px; padding:8px 10px; border-left:3px solid #B0443F; background:#F4F6F9; margin:4px 0; }
.leyenda { margin: 4px 0 12px; }
.leyenda-item { font-size:13px; margin-right:18px; display:inline-flex; align-items:center; }
.leyenda-item i { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; }
.pasos { display:flex; gap:6px; margin-bottom:8px; }
.pasos i { flex:1; height:6px; background:#C9D2DC; display:block; }
.pasos i.on { background:#15233A; }
table.tabla { border-collapse:separate; border-spacing:4px; font-size:14px; width:100%; }
table.tabla th { font-weight:500; color:#4A5B70; font-size:12px; padding:4px 6px; text-align:center; }
table.tabla td { text-align:center; padding:9px 6px; border-radius:2px; background:#DDE4EC; font-variant-numeric: tabular-nums; }
div[data-testid="stTabs"] button[role="tab"] { font-size:15px; color:#4A5B70; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#15233A; font-weight:600; }
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] { background:#15233A; }
div[data-testid="stTextInput"] div[data-baseweb="base-input"], div[data-testid="stTextInput"] div[data-baseweb="input"] { background:#F4F6F9 !important; border:1px solid #8C9BAD !important; border-radius:2px !important; }
div[data-testid="stTextInput"] input { background:#F4F6F9 !important; color:#15233A !important; -webkit-text-fill-color:#15233A !important; caret-color:#15233A; }
div[data-testid="stTextInput"] input::placeholder { color:#8C9BAD !important; -webkit-text-fill-color:#8C9BAD !important; opacity:1; }
div[data-testid="stTextInput"] label p { color:#15233A !important; font-size:13px !important; }
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-secondary"] { border-radius:2px !important; }
button[data-testid="stBaseButton-primary"] { background:#15233A !important; border:1px solid #15233A !important; color:#F4F6F9 !important; }
button[data-testid="stBaseButton-primary"]:hover { background:#4A5B70 !important; border-color:#4A5B70 !important; }
button[data-testid="stBaseButton-secondary"] { background:#F4F6F9 !important; border:1px solid #8C9BAD !important; color:#15233A !important; }
div[role="dialog"], div[data-testid="stDialog"] div[role="dialog"] { background:#F4F6F9 !important; color:#15233A !important; }
div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3, div[role="dialog"] p, div[role="dialog"] label, div[role="dialog"] span { color:#15233A; }
div[role="dialog"] .mut { color:#4A5B70 !important; }
div[role="dialog"] button[aria-label="Close"] { color:#15233A !important; }
</style>
"""

LOGO = """<svg width="40" height="40" viewBox="0 0 36 36" aria-hidden="true">
<path d="M11 10 L16 6 M11 10 L16 18 M11 26 L16 18 M11 26 L16 30 M21 6 L25 16 M21 18 L25 18 M21 30 L25 20" stroke="#4A5B70" stroke-width="1.4" fill="none"/>
<circle cx="8" cy="10" r="3.6" fill="#B7791F"/><circle cx="8" cy="26" r="3.6" fill="#B7791F"/>
<circle cx="18" cy="6" r="3" fill="#15233A"/><circle cx="18" cy="18" r="3" fill="#15233A"/><circle cx="18" cy="30" r="3" fill="#15233A"/>
<circle cx="28" cy="18" r="4.2" fill="#B7791F"/></svg>"""


def fmt(x, dec=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{dec}f}".replace(".", ",")


def etiqueta(nombre):
    return ETIQUETAS.get(nombre, nombre.replace("_", " ").capitalize())


def marcador(c):
    """Texto de ejemplo dentro del campo vacío."""
    if c["tipo"] == "num":
        d = 0 if c["entero"] else 2
        return f"Entre {fmt(c['min'], d)} y {fmt(c['max'], d)}"
    return f"Ej.: {c['valores'][0]}" if c.get("valores") else "Escriba un texto"


def requisitos(c):
    """Línea bajo el campo con lo que el dataset espera."""
    if c["tipo"] == "num":
        d = 0 if c["entero"] else 2
        partes = ["Número"] + (["entero"] if c["entero"] else []) + (["no negativo"] if c["no_negativo"] else [])
        return " · ".join(partes) + f" · rango observado {fmt(c['min'], d)} a {fmt(c['max'], d)}"
    n = c["n_valores"]
    base = f"{n} {'valor conocido' if n == 1 else 'valores conocidos'}"
    if categoria_numerica(c):
        return f"Solo números · {base}"
    return "Solo " + texto_admitido(c) + f" · {base}"


def clave(ds, nombre):
    return f"{ds}_{nombre}"


@st.cache_data(show_spinner=False)
def cargar_campos(ds):
    return json.loads((I.AUX / f"campos_{ds}.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def cargar_metricas(ds):
    ajuste = json.loads((CARPETA / f"ajuste_{ds}.json").read_text(encoding="utf-8"))
    top = json.loads((CARPETA / f"top3_{ds}.json").read_text(encoding="utf-8"))
    return ajuste, top


# Campos donde un valor negativo tiene sentido real (p. ej. saldo a favor). En el resto se bloquea.
NEGATIVOS_CON_SENTIDO = {"Saldo_Actual"}
MAX_DIGITOS = 15            # único límite duro numérico: evita entradas absurdas; fuera de rango solo avisa
LARGO_MAX_TEXTO = 60
RE_ENTERO = re.compile(r"^-?\d+$")
RE_DECIMAL = re.compile(r"^-?\d+([.,]\d+)?$")


def nombre_campo(c):
    return c["nombre"]


def categoria_numerica(c):
    """Categorías codificadas con números (p. ej. '0.0', '1.0'): se ingresan como número."""
    return bool(c.get("valores")) and all(RE_DECIMAL.match(v) for v in c["valores"])


def simbolos_permitidos(c):
    """Símbolos (no letras ni espacios) que aparecen en los valores reales del campo."""
    return {ch for v in c.get("valores", []) for ch in v if not ch.isalnum() and ch != " "}


def texto_admitido(c):
    """Descripción corta de lo que admite un campo de texto, p. ej. «letras, dígitos y los símbolos - + <»."""
    simb = sorted(simbolos_permitidos(c))
    digitos = any(ch.isdigit() for v in c["valores"] for ch in v)
    partes = ["letras"] + (["dígitos"] if digitos else [])
    texto = ", ".join(partes)
    return texto + (f" y los símbolos {' '.join(simb)}" if simb else "")


def revisar(c, bruto):
    """Devuelve (error, None, valor limpio) de un campo. Solo los errores de formato bloquean la clasificación;
    los valores fuera de lo habitual se aceptan sin mensaje, para probar la red con casos atípicos."""
    et = etiqueta(c["nombre"])
    bruto = bruto or ""
    t = bruto.strip()
    if t == "":
        return f"{et}: falta el dato. Escriba un valor.", None, None
    if bruto != t:
        return f"{et}: no puede empezar ni terminar con espacios.", None, None
    if "  " in t:
        return f"{et}: no puede tener dos espacios seguidos.", None, None

    if c["tipo"] == "num":
        patron = RE_ENTERO if c["entero"] else RE_DECIMAL
        if not patron.match(t):
            if c["entero"]:
                return f"{et}: solo admite un número entero (dígitos, y «-» al inicio si aplica).", None, None
            return f"{et}: solo admite números (dígitos, «-» al inicio y un punto o coma decimal).", None, None
        if len(re.sub(r"\D", "", t)) > MAX_DIGITOS:
            return f"{et}: máximo {MAX_DIGITOS} dígitos.", None, None
        x = float(t.replace(",", "."))
        if x < 0 and c["no_negativo"] and nombre_campo(c) not in NEGATIVOS_CON_SENTIDO:
            return f"{et}: no puede ser negativo. Un valor negativo no tiene sentido en este campo.", None, None
        return None, None, x

    # Categorías
    conocidos = {v.upper(): v for v in c["valores"]}
    if categoria_numerica(c):
        if not RE_DECIMAL.match(t):
            return f"{et}: solo admite números.", None, None
        f = float(t.replace(",", "."))
        for v in c["valores"]:
            if float(v) == f:
                return None, None, v
        return None, None, t
    if len(t) > LARGO_MAX_TEXTO:
        return f"{et}: máximo {LARGO_MAX_TEXTO} caracteres.", None, None
    permitidos = simbolos_permitidos(c)
    if any(ch.isdigit() for v in c["valores"] for ch in v):
        permitidos |= set("0123456789")
    malos = sorted({ch for ch in t if not (ch.isalpha() or ch == " " or ch in permitidos)})
    if malos:
        return (f"{et}: solo admite {texto_admitido(c)}. No se permite «{''.join(malos)}»."), None, None
    if t.upper() in conocidos:
        return None, None, conocidos[t.upper()]
    return None, None, t


def _poner(d, k, v):
    if v:
        d[k] = v
    else:
        d.pop(k, None)


def _al_cambiar(ds, c):
    """Se ejecuta al cambiar un campo: el error aparece en cuanto se escribe, no al final."""
    err, aviso, _ = revisar(c, st.session_state.get(clave(ds, c["nombre"]), ""))
    _poner(st.session_state.setdefault(f"err_{ds}", {}), c["nombre"], err)
    _poner(st.session_state.setdefault(f"av_{ds}", {}), c["nombre"], aviso)


def _guardar_paso(ds, grupo):
    datos = st.session_state.setdefault(f"datos_{ds}", {})
    for c in grupo:
        datos[c["nombre"]] = st.session_state.get(clave(ds, c["nombre"]), datos.get(c["nombre"], ""))


def barras(clases, probs):
    filas = []
    for i, (c, p) in enumerate(zip(clases, probs)):
        filas.append(f'<div style="margin:8px 0"><div style="display:flex;justify-content:space-between;font-size:13px">'
                     f'<span>{frontera.nombre(c)}</span><span class="num">{fmt(p * 100, 1)} %</span></div>'
                     f'<div class="barra"><i style="width:{p * 100:.1f}%;background:{frontera.COLORES[i]}"></i></div></div>')
    return "".join(filas)


def render_resultado(res):
    clase = frontera.nombre(res["clase"])
    duda = ('<span class="chip">En zona de duda</span>' if res.get("zona") else '<span class="chip">Fuera de zona de duda</span>') \
        if res.get("zona") is not None else ""
    return (f'<div class="tarjeta"><div class="mut" style="font-size:13px">Clase predicha</div>'
            f'<div class="disp" style="font-size:36px;line-height:1.1">{clase}</div>'
            f'<div style="margin:8px 0">{duda}</div>{barras(res["clases"], res["probs"])}</div>')


def seccion_dashboard(ds):
    ajuste, top = cargar_metricas(ds)
    p, v = ajuste["prueba"], ajuste["validacion"]
    cfg = ajuste["config_elegida"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="tarjeta oscura"><div class="mut">AUC de prueba</div><div class="valor">{fmt(p["auc_macro_ovr"])}</div>'
                f'<div class="nota">Uno contra el resto, promedio macro</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="tarjeta"><div class="mut">Exactitud balanceada, prueba</div><div class="valor">{fmt(p["exactitud_balanceada"])}</div>'
                f'<div class="nota">Validación: {fmt(v["exactitud_balanceada"])}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="tarjeta"><div class="mut">Exactitud, prueba</div><div class="valor">{fmt(p["exactitud"])}</div>'
                f'<div class="nota">Sobre el conjunto de prueba</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="tarjeta"><div class="mut">CV balanceada (k = 5)</div><div class="valor">{fmt(ajuste["cv_promedio"])}</div>'
                f'<div class="nota">Época de parada: {ajuste["epoca_final"]}</div></div>', unsafe_allow_html=True)
    capas = " → ".join(str(n) for n in cfg["capas_ocultas"])
    st.markdown(f'<div class="mut" style="font-size:13px;margin:10px 0 14px">Arquitectura elegida: {capas} neuronas, '
                f'α (L2) {fmt(cfg["alpha"], 1)}, lote {cfg["tamano_lote"]}, tasa {fmt(cfg["tasa_aprendizaje"], 4)}, '
                f'sobremuestreo {"sí" if cfg["sobremuestreo"] else "no"}.</div>', unsafe_allow_html=True)
    filas = []
    for c in top["configuraciones"]:
        marca = '<span class="chip">Elegida</span>' if c["elegida"] else ""
        estilo = ' style="font-weight:600"' if c["elegida"] else ""
        filas.append(f'<tr{estilo}><td>{c["posicion"]}</td><td style="text-align:left">{" → ".join(str(n) for n in c["capas_ocultas"])} neuronas</td>'
                     f'<td>{fmt(c["alpha_l2"], 1)}</td><td>{c["tamano_lote"]}</td><td>{fmt(c["cv_promedio"])}</td>'
                     f'<td>{fmt(min(c["cv_por_semilla"]))} – {fmt(max(c["cv_por_semilla"]))}</td><td>{marca}</td></tr>')
    st.markdown('<div class="disp" style="font-size:24px;margin:6px 0">Tres mejores configuraciones</div>', unsafe_allow_html=True)
    st.markdown('<table class="tabla"><thead><tr><th>#</th><th style="text-align:left">Arquitectura</th><th>α</th><th>Lote</th>'
                '<th>CV balanceada</th><th>Rango por semilla</th><th></th></tr></thead><tbody>' + "".join(filas) + "</tbody></table>",
                unsafe_allow_html=True)


def _clasificar(ds, campos, grupos):
    """Valida todos los campos. Si hay errores, lleva al primer paso con error; si no, clasifica."""
    datos = st.session_state.get(f"datos_{ds}", {})
    errs = st.session_state.setdefault(f"err_{ds}", {})
    avs = st.session_state.setdefault(f"av_{ds}", {})
    primero, limpio = None, {}
    for gi, grupo in enumerate(grupos):
        for c in grupo:
            err, aviso, valor = revisar(c, datos.get(c["nombre"], ""))
            _poner(errs, c["nombre"], err)
            _poner(avs, c["nombre"], aviso)
            if err and primero is None:
                primero = gi
            limpio[c["nombre"]] = valor
    if primero is not None:
        st.session_state[f"paso_{ds}"] = primero
        st.rerun(scope="fragment")
    fila = pd.DataFrame([limpio])
    clases, pr = I.predecir(ds, fila)
    zona = I.zona_duda(ds, fila)
    res = {"clase": clases[0], "probs": pr[0].tolist(), "clases": I.cargar(ds)["aux"]["clases"], "zona": zona,
           "nn": I.entrada_red(ds, fila).to_numpy(float)[0].tolist(), "avisos": list(avs.values())}
    st.session_state[f"res_{ds}"] = res
    hist = st.session_state.setdefault(f"hist_{ds}", [])
    hist.insert(0, {"hora": datetime.now().strftime("%H:%M:%S"), "clase": clases[0],
                    "confianza": f"{max(res['probs']) * 100:.1f} %".replace(".", ","),
                    "zona_duda": "sí" if zona else "no" if zona is not None else "—"})
    st.session_state[f"paso_{ds}"] = 0
    st.rerun()


@st.dialog("Nueva clasificación", width="large")
def asistente(ds):
    _, centro, _ = st.columns([1, 5, 1])
    with centro:
        campos = cargar_campos(ds)
        grupos = [campos[i:i + CAMPOS_POR_PASO] for i in range(0, len(campos), CAMPOS_POR_PASO)]
        paso = st.session_state.get(f"paso_{ds}", 0)
        datos = st.session_state.setdefault(f"datos_{ds}", {})
        errs = st.session_state.setdefault(f"err_{ds}", {})
        avs = st.session_state.setdefault(f"av_{ds}", {})
        grupo, ultimo = grupos[paso], paso == len(grupos) - 1
        ini = paso * CAMPOS_POR_PASO

        barra = "".join(f'<i class="{"on" if i <= paso else ""}"></i>' for i in range(len(grupos)))
        st.markdown(f'<div class="pasos">{barra}</div><div class="mut" style="font-size:13px;margin-bottom:12px">'
                    f'Paso {paso + 1} de {len(grupos)} · campos {ini + 1} a {ini + len(grupo)} de {len(campos)}</div>',
                    unsafe_allow_html=True)

        cols = st.columns(2)
        for k, c in enumerate(grupo):
            with cols[k % 2]:
                st.text_input(etiqueta(c["nombre"]), value=datos.get(c["nombre"], ""), key=clave(ds, c["nombre"]),
                              placeholder=marcador(c), on_change=_al_cambiar, args=(ds, c))
                st.markdown(f'<div class="mut" style="font-size:12px;margin:-6px 0 4px">{requisitos(c)}</div>', unsafe_allow_html=True)
                if c["nombre"] in errs:
                    st.markdown(f'<div class="error">{errs[c["nombre"]]}</div>', unsafe_allow_html=True)
                elif c["nombre"] in avs:
                    st.markdown(f'<div class="aviso">{avs[c["nombre"]]}</div>', unsafe_allow_html=True)

        izq, der = st.columns(2)
        if paso > 0 and izq.button("Anterior", key=f"ant_{ds}"):
            _guardar_paso(ds, grupo)
            st.session_state[f"paso_{ds}"] = paso - 1
            st.rerun(scope="fragment")
        if der.button("Clasificar operación" if ultimo else "Siguiente", key=f"sig_{ds}", type="primary"):
            _guardar_paso(ds, grupo)
            for c in grupo:
                err, aviso, _ = revisar(c, datos.get(c["nombre"], ""))
                _poner(errs, c["nombre"], err)
                _poner(avs, c["nombre"], aviso)
            if any(c["nombre"] in errs for c in grupo):
                st.rerun(scope="fragment")
            if not ultimo:
                st.session_state[f"paso_{ds}"] = paso + 1
                st.rerun(scope="fragment")
            _clasificar(ds, campos, grupos)


def seccion_formulario(ds):
    campos = cargar_campos(ds)
    n_pasos = -(-len(campos) // CAMPOS_POR_PASO)
    res = st.session_state.get(f"res_{ds}")

    st.markdown('<div class="disp" style="font-size:24px;margin:18px 0 4px">Clasificar una operación</div>', unsafe_allow_html=True)
    izq, der = st.columns([1, 1.3])
    with izq:
        st.markdown(f'<div class="mut" style="font-size:14px;margin-bottom:12px">Ingrese los {len(campos)} datos de la operación '
                    f'en {n_pasos} pasos. Cada campo indica el formato que espera el dataset y muestra el error '
                    'en cuanto se escribe. Los valores fuera de lo habitual se aceptan.</div>', unsafe_allow_html=True)
        if st.button("Nueva clasificación", key=f"abrir_{ds}", type="primary"):
            for prefijo in ("paso", "datos", "err", "av"):
                st.session_state[f"{prefijo}_{ds}"] = 0 if prefijo == "paso" else {}
            asistente(ds)
    with der:
        if res:
            st.markdown(render_resultado(res), unsafe_allow_html=True)
            if res.get("avisos"):
                st.markdown("".join(f'<div class="aviso">{a}</div>' for a in res["avisos"]), unsafe_allow_html=True)
        else:
            st.markdown('<div class="tarjeta mut">Aún no hay clasificaciones en esta sesión.</div>', unsafe_allow_html=True)

    hist = st.session_state.get(f"hist_{ds}")
    if hist:
        st.markdown('<div class="disp" style="font-size:22px;margin:22px 0 6px">Operaciones clasificadas en esta sesión</div>', unsafe_allow_html=True)
        filas = "".join(f'<tr><td>{h["hora"]}</td><td style="text-align:left">{frontera.nombre(h["clase"])}</td>'
                        f'<td>{h["confianza"]}</td><td>{h["zona_duda"]}</td></tr>' for h in hist)
        st.markdown('<table class="tabla"><thead><tr><th>Hora</th><th style="text-align:left">Clase predicha</th><th>Confianza</th>'
                    '<th>Zona de duda</th></tr></thead><tbody>' + filas + "</tbody></table>", unsafe_allow_html=True)

    nn = res.get("nn") if res else None
    st.markdown('<div class="disp" style="font-size:24px;margin:26px 0 4px">Frontera de decisión</div>', unsafe_allow_html=True)
    st.markdown('<div class="mut" style="font-size:13px;margin-bottom:8px">Cada color es una clase. Las líneas oscuras son el borde '
                'donde la red cambia de clase. Los puntos son casos de validación en su clase real. Si clasificó un dato, aparece '
                'marcado con su clase predicha.</div>', unsafe_allow_html=True)
    st.image(frontera.figura(ds, nn=nn, pred=res.get("clase") if nn else None))
    st.markdown('<div class="mut" style="font-size:12px;margin-top:6px">Las dos componentes explican solo una parte de la varianza: '
                'una zona que parece cruzar una frontera puede no cruzarla en todas las variables.</div>', unsafe_allow_html=True)


def encabezado():
    st.markdown(f'<div class="navbar"><div class="marca">{LOGO}<div><div class="disp" style="font-size:22px;line-height:1">'
                f'Red neuronal</div><div class="mut" style="font-size:12px;margin-top:4px">Cartera de crédito · cuatro modelos '
                f'con mayor AUC de prueba</div></div></div><div><span class="chip">Entrenamiento local · CPU</span></div></div>',
                unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Red neuronal · clasificación", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    encabezado()
    tabs = st.tabs([nombre for _, nombre in TOP4])
    for tab, (ds, _) in zip(tabs, TOP4):
        with tab:
            seccion_dashboard(ds)
            seccion_formulario(ds)


if __name__ == "__main__":
    main()
