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
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent / "nucleo"))

import frontera  # noqa: E402
import ficha_red  # noqa: E402
import inferencia as I  # noqa: E402
import cascada  # noqa: E402
import garantia  # noqa: E402
import visual_red  # noqa: E402

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
    if es_booleano(c):
        return "Seleccione SI o NO"
    if c["tipo"] == "num":
        minimo, tope = limites_numero(c)
        partes = ["Número"] + (["entero"] if c["entero"] else ["con decimales"]) + \
            (["no negativo"] if minimo == 0 else []) + (["opcional (vac\u00edo = 0)"] if c.get("opcional") else []) + ([c["ayuda"]] if c.get("ayuda") else [])
        return " · ".join(partes) + f" · hasta {len(str(tope))} dígitos (máximo {tope})"
    n = c["n_valores"]
    base = f"{n} {'valor conocido' if n == 1 else 'valores conocidos'}"
    if categoria_numerica(c):
        return f"Solo números · {base}"
    return "Solo " + texto_admitido(c) + f" · {base}"


def clave(ds, nombre):
    return f"{ds}_{nombre}"


@st.cache_data(show_spinner=False)
def cargar_campos(ds):
    campos = json.loads((I.AUX / f"campos_{ds}.json").read_text(encoding="utf-8"))
    for c in campos:
        if c["tipo"] == "num" and I.es_monetario(c["nombre"]):
            c["entero"] = False
    return campos


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


VALORES_BOOLEANOS = {"0", "1", "0.0", "1.0"}


def es_booleano(c):
    """Campo de dos valores 0/1 (sí/no): se ingresa con SI o NO, sin teclear números."""
    vals = c.get("valores") or []
    return c["tipo"] == "cat" and bool(vals) and set(vals) <= VALORES_BOOLEANOS


def limites_numero(c):
    """Rango que acepta el campo: tantos dígitos como el máximo observado y signo según la regla del campo."""
    tope = 10 ** len(str(int(abs(c["max"])))) - 1
    minimo = -tope if (not c["no_negativo"] or nombre_campo(c) in NEGATIVOS_CON_SENTIDO) else 0
    return minimo, tope


def _revisar_booleano(c, et, bruto):
    if bruto in (None, ""):
        return f"{et}: falta el dato. Seleccione SI o NO.", None, None
    if bruto not in ("SI", "NO"):
        return f"{et}: solo admite SI o NO.", None, None
    objetivo = 1.0 if bruto == "SI" else 0.0
    return None, None, next((v for v in c["valores"] if float(v) == objetivo), "1" if objetivo else "0")


def _revisar_numero(c, et, bruto):
    if bruto is None or bruto == "":
        if c.get("opcional"):
            return None, None, 0.0
        return f"{et}: falta el dato. {c.get('ayuda') or 'Escriba un valor.'}", None, None
    if isinstance(bruto, str):
        return f"{et}: solo admite números.", None, None
    x = float(bruto)
    minimo, tope = limites_numero(c)
    if c["entero"] and x != int(x):
        return f"{et}: solo admite un número entero.", None, None
    if x < minimo:
        return (f"{et}: no puede ser negativo. Un valor negativo no tiene sentido en este campo.", None, None) \
            if minimo == 0 else (f"{et}: el mínimo es {minimo}.", None, None)
    if x > tope:
        return f"{et}: el máximo es {tope}.", None, None
    return None, (c.get("aviso_cero") if x == 0 else None), x


def revisar(c, bruto):
    """Devuelve (error, None, valor limpio) de un campo. Solo los errores de formato bloquean la clasificación;
    los valores fuera de lo habitual se aceptan sin mensaje, para probar la red con casos atípicos."""
    et = etiqueta(c["nombre"])
    if es_booleano(c):
        return _revisar_booleano(c, et, bruto)
    if c["tipo"] == "num":
        return _revisar_numero(c, et, bruto)
    bruto = bruto or ""
    t = bruto.strip()
    if t == "":
        return f"{et}: falta el dato. Escriba un valor.", None, None
    if bruto != t:
        return f"{et}: no puede empezar ni terminar con espacios.", None, None
    if "  " in t:
        return f"{et}: no puede tener dos espacios seguidos.", None, None

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


def estado_cascada(res):
    """Banner of the two-step cascade (out-of-range filter, then ensemble ambiguity). Empty when not available."""
    c = res.get("cascada")
    if not c:
        return ""
    if c["estado"] == cascada.AUTOMATICO:
        clase = frontera.nombre(res["clases"][c["clase_idx"]])
        return f'<div class="chip">Cascada: autom&aacute;tico, clase del ensamble {clase}</div>'
    motivo = "fuera de rango" if c["motivo"] == cascada.FUERA_DE_RANGO else "ambig&uuml;edad del ensamble"
    return (f'<div class="aviso"><b>REVISI&Oacute;N MANUAL</b>: {motivo}. Entrop&iacute;a {fmt(c["entropia"], 2)}, '
            f'distancia al entrenamiento {fmt(c["distancia"], 2)}.</div>')


def render_resultado(res):
    clase = frontera.nombre(res["clase"])
    duda = ('<span class="chip">En zona de duda</span>' if res.get("zona") else '<span class="chip">Fuera de zona de duda</span>') \
        if res.get("zona") is not None else ""
    return (f'<div class="tarjeta"><div class="mut" style="font-size:13px">Clase predicha</div>'
            f'<div class="disp" style="font-size:36px;line-height:1.1">{clase}</div>'
            f'<div style="margin:8px 0">{duda}</div>{estado_cascada(res)}{barras(res["clases"], res["probs"])}</div>')


@st.cache_data(show_spinner=False)
def cargar_ficha(ds):
    return ficha_red.cargar_ficha(ds)


def _texto_config(f):
    c = f["config"]
    capas = " → ".join(str(n) for n in f["capas"])
    return (f'Capas: <b>{capas}</b> (entradas → oculta → salida) · α (L2) <b>{fmt(c["alpha"], 1)}</b> · lote <b>{c["tamano_lote"]}</b> · '
            f'tasa de aprendizaje <b>{fmt(c["tasa_aprendizaje"], 4)}</b> · sobremuestreo <b>{"sí" if c["sobremuestreo"] else "no"}</b> · '
            'activación ReLU, salida softmax')


def _resumen_config(f):
    return f'<div class="mut" style="font-size:13px;margin:4px 0 10px">{_texto_config(f)}</div>'


def _tabla_confusion(f):
    nombres = [frontera.nombre(c) for c in f["clases"]]
    cabecera = "".join(f"<th>{n}</th>" for n in nombres)
    filas = []
    for k, fila in enumerate(f["confusion"]):
        celdas = "".join(f'<td style="{"font-weight:600;background:#C9D2DC" if j == k else ""}">{v}</td>' for j, v in enumerate(fila))
        filas.append(f'<tr><td style="text-align:left">{nombres[k]}</td>{celdas}<td>{fmt(f["recall"][k], 2)}</td></tr>')
    precision = "".join(f'<td>{fmt(p, 2)}</td>' for p in f["precision"])
    filas.append(f'<tr style="color:#4A5B70"><td style="text-align:left">Precisión</td>{precision}<td></td></tr>')
    return ('<table class="tabla"><thead><tr><th style="text-align:left">Real \\ predicha</th>' + cabecera +
            '<th>Recall</th></tr></thead><tbody>' + "".join(filas) + "</tbody></table>")


def mostrar_html(html, alto):
    """st.iframe replaces components.html, which Streamlit removes; keep the old call for older versions."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=alto)
    else:
        components.html(html, height=alto)


def _pestana_diagrama(f):
    if f["nota"]:
        st.markdown(f'<div class="aviso">{f["nota"]}</div>', unsafe_allow_html=True)
    st.markdown(_resumen_config(f), unsafe_allow_html=True)
    html = visual_red.html_lienzo(f["capas"], f["coefs"], modo="final", entradas=f["entradas"], clases=f["clases"], alto=520,
                                  umbral=ficha_red.UMBRAL_INICIAL, separacion=ficha_red.separacion_capas(f["capas"]))
    mostrar_html(html, 530)
    st.markdown('<div class="mut" style="font-size:12px">Arrastre para mover, use la rueda para acercar y el control '
                '"Mostrar desde" para ocultar las conexiones más débiles. Pase el cursor sobre una neurona para ver su nombre.</div>',
                unsafe_allow_html=True)


def pct(x, dec=2):
    return f"{fmt(x * 100, dec)} %"


def pp(x, dec=2):
    """Signed difference in percentage points, with the real minus sign so it reads well in a table."""
    texto = fmt(abs(x) * 100, dec)
    return f'{"+" if x >= 0 else "−"}{texto} pp'


def _estado_limite(cumple, limite):
    if cumple:
        return f'<span class="chip" style="border-color:#2F6B4F;color:#2F6B4F">Cumple el límite de {limite} %</span>'
    return f'<span class="chip" style="border-color:#B0443F;color:#B0443F">No cumple el límite de {limite} %</span>'


def _estado_limite_margen(cumple):
    if cumple:
        return ('<span class="chip" style="border-color:#2F6B4F;color:#2F6B4F">La prueba alcanza el margen de ± 2 pp</span>')
    return ('<span class="chip" style="border-color:#B0443F;color:#B0443F">La prueba no alcanza el margen de ± 2 pp; '
            'el margen real es el que se muestra arriba</span>')


def _seccion_entrenamiento(f):
    e = f["entrenamiento"]
    st.markdown('<div class="disp" style="font-size:22px;margin:4px 0 6px">Entrenamiento y generalización</div>', unsafe_allow_html=True)
    if e is None:
        st.markdown('<div class="aviso">No se encontraron los datos de entrenamiento (X_train_nn.csv y compañeros) junto a este programa; '
                    'no se puede calcular el error de entrenamiento ni la brecha.</div>', unsafe_allow_html=True)
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="tarjeta"><div class="mut">Error de entrenamiento</div><div class="valor">{pct(e["error"]["train"])}</div>'
                f'<div class="nota">Sobre las filas originales de train</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="tarjeta"><div class="mut">Error de validación</div><div class="valor">{pct(e["error"]["val"])}</div>'
                f'<div class="nota">Brecha con train: {pp(e["brecha_val"])}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="tarjeta"><div class="mut">Error de prueba</div><div class="valor">{pct(e["error"]["test"])}</div>'
                f'<div class="nota">Solo se mide, no decide nada</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="tarjeta oscura"><div class="mut">Brecha |prueba − entrenamiento|</div>'
                f'<div class="valor">{fmt(e["brecha_abs"] * 100, 2)} pp</div>'
                f'<div class="nota">Prueba − entrenamiento: {pp(e["brecha"])}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="margin:8px 0 2px">{_estado_limite(e["cumple_5"], 5)} {_estado_limite(e["cumple_3"], 3)}</div>'
                '<div class="mut" style="font-size:12px;margin-bottom:10px">Error = 1 − exactitud. La brecha es la diferencia absoluta entre el '
                'error de prueba y el de entrenamiento, en puntos porcentuales (pp); se compara con 5 %, y con 3 % que es el criterio más '
                'estricto de la documentación. Una brecha grande indica sobreajuste; una brecha negativa, que la prueba resultó más fácil '
                'que el entrenamiento.</div>', unsafe_allow_html=True)
    filas = []
    for nombre, clave, formato in (("Error", "error", pct), ("Exactitud balanceada", "exactitud_balanceada", fmt),
                                   ("Macro F1", "macro_f1", fmt), ("Pérdida (log-loss)", "perdida", fmt)):
        r = e[clave]
        filas.append(f'<tr><td style="text-align:left">{nombre}</td><td>{formato(r["train"])}</td><td>{formato(r["val"])}</td>'
                     f'<td>{formato(r["test"])}</td></tr>')
    st.markdown('<table class="tabla"><thead><tr><th style="text-align:left">Medida</th><th>Entrenamiento</th><th>Validación</th><th>Prueba</th>'
                '</tr></thead><tbody>' + "".join(filas) + "</tbody></table>"
                '<div class="mut" style="font-size:12px">Calculado con los pesos guardados, sin calibrar. El entrenamiento usó sobremuestreo, '
                'pero aquí se mide sobre las filas originales.</div>', unsafe_allow_html=True)
    h = pd.DataFrame(e["historial"]).set_index("epoca")
    st.markdown(f'<div class="disp" style="font-size:20px;margin:16px 0 4px">Curva de entrenamiento</div>'
                f'<div class="mut" style="font-size:12px;margin-bottom:6px">Se entrenaron {len(h)} épocas y se conservó la época '
                f'{f["metricas"]["epoca_final"]}, la de mejor exactitud balanceada de validación (parada temprana).</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    g1.line_chart(h[["perdida_train"]].rename(columns={"perdida_train": "Pérdida de entrenamiento"}), height=220)
    g2.line_chart(h[["exactitud_balanceada_val"]].rename(columns={"exactitud_balanceada_val": "Exactitud balanceada de validación"}), height=220)


def _pestana_metricas(f):
    m, p, v = f["metricas"], f["metricas"]["prueba"], f["metricas"]["validacion"]
    _seccion_entrenamiento(f)
    st.markdown('<div class="disp" style="font-size:22px;margin:18px 0 6px">Rendimiento</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="tarjeta oscura"><div class="mut">AUC de prueba</div><div class="valor">{fmt(p["auc_macro_ovr"])}</div>'
                f'<div class="nota">Validación: {fmt(v["auc_macro_ovr"])}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="tarjeta"><div class="mut">Exactitud balanceada</div><div class="valor">{fmt(p["exactitud_balanceada"])}</div>'
                f'<div class="nota">Validación: {fmt(v["exactitud_balanceada"])}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="tarjeta"><div class="mut">Exactitud</div><div class="valor">{fmt(p["exactitud"])}</div>'
                f'<div class="nota">Validación: {fmt(v["exactitud"])}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="tarjeta"><div class="mut">CV balanceada (k = 5)</div><div class="valor">{fmt(m["cv_promedio"])}</div>'
                f'<div class="nota">Rango por semilla: {fmt(min(m["cv_por_semilla"]))} – {fmt(max(m["cv_por_semilla"]))}</div></div>',
                unsafe_allow_html=True)
    if f["nota"]:
        st.markdown(f'<div class="aviso" style="margin-top:10px">{f["nota"]}</div>', unsafe_allow_html=True)
    if f["temperatura"] is not None:
        cal = (f'Calibración por temperatura aplicada: T = {fmt(f["temperatura"])}. Cambia la confianza mostrada, no la clase predicha.')
    else:
        cal = 'Sin calibración por temperatura: no mejoró en prueba. La confianza mostrada es la que entrega la red.'
    st.markdown(f'<div class="mut" style="font-size:13px;margin:12px 0 4px">{_texto_config(f)}'
                f'<br>Época final: {m["epoca_final"]} (entrenó {m["epocas_entrenadas"]} épocas con parada temprana). {cal}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="disp" style="font-size:22px;margin:14px 0 4px">Matriz de confusión en prueba</div>', unsafe_allow_html=True)
    st.markdown(_tabla_confusion(f), unsafe_allow_html=True)
    st.markdown('<div class="mut" style="font-size:12px">Las filas son la clase real y las columnas la predicha. '
                'Recall: de los casos reales de la fila, cuántos acertó. Precisión: de los casos que predijo en la columna, cuántos eran de esa clase.</div>',
                unsafe_allow_html=True)


def _bloque_pregunta(p):
    """Question the network answers (as a question), its source in the design letter, and how the network answers it."""
    if not p or not p.get("pregunta"):
        return
    st.markdown(f'<div class="tarjeta" style="margin:4px 0 14px">'
                f'<div class="mut" style="font-size:12px">{p["origen"]}</div>'
                f'<div class="disp" style="font-size:20px;margin:4px 0 8px">{p["pregunta"]}</div>'
                f'<div class="mut" style="font-size:12px">Cómo responde la red</div>'
                f'<div style="font-size:14px;margin-top:2px">{p["como_responde"]}</div></div>',
                unsafe_allow_html=True)


def _pestana_justificacion(f):
    j = f["justificacion"]
    _bloque_pregunta(j["pregunta"])
    h, d, x = j["hoeffding"], j["dimension"], j["extra"]
    st.markdown('<div class="disp" style="font-size:22px;margin:14px 0 4px">Tamaño de la prueba y margen de error</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="tarjeta oscura"><div class="mut">Filas de prueba</div><div class="valor">{j["n_prueba"]:,}</div>'
                f'<div class="nota">Conjunto con el que se mide el error</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="tarjeta"><div class="mut">Margen de error (Hoeffding, 95 %)</div>'
                f'<div class="valor">± {pct(h["margen_error"], 2)}</div>'
                f'<div class="nota">El error real está en el error medido ± este margen</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="tarjeta"><div class="mut">Filas para ± 2 pp / ± 1 pp</div>'
                f'<div class="valor">{h["n_para_2_pp"]:,} / {h["n_para_1_pp"]:,}</div>'
                f'<div class="nota">Mínimo de filas de prueba con ese margen</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="mut" style="font-size:12px;margin-top:6px">Desigualdad de Hoeffding: para n filas independientes, '
                'la probabilidad de que el error medido se aleje del real en más de ε es como mucho 2·exp(−2·n·ε²). '
                'Con confianza de 95 %, ε = √(ln(2/0,05) / (2n)). Supone filas independientes; los cortes temporales '
                'solo lo aproximan.</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="margin:8px 0 2px">{_estado_limite_margen(h["cumple_2_pp"])}</div>', unsafe_allow_html=True)
    vc = j["validacion_cruzada"]
    st.markdown('<div class="disp" style="font-size:22px;margin:14px 0 4px">Validación cruzada de la búsqueda</div>', unsafe_allow_html=True)
    c_k, c_f = st.columns(2)
    c_k.markdown(f'<div class="tarjeta"><div class="mut">Pliegues (k)</div><div class="valor">{vc["k"]}</div>'
                 f'<div class="nota">Validación cruzada {vc["tipo"]}</div></div>', unsafe_allow_html=True)
    c_f.markdown(f'<div class="tarjeta"><div class="mut">Filas de validación por pliegue</div>'
                 f'<div class="valor">{vc["filas_por_pliegue"]:,.0f}</div>'
                 f'<div class="nota">Margen de error por pliegue: ± {pct(vc["margen_por_pliegue"], 2)}</div></div>',
                 unsafe_allow_html=True)
    st.markdown('<div class="mut" style="font-size:12px;margin-top:6px">Cada pliegue se usa una vez para validar y el resto para '
                'entrenar. Las cifras de la búsqueda son el promedio de los cinco pliegues: cada uno tiene menos filas que la prueba, '
                'y el promedio se apoya en todo el conjunto de entrenamiento, pero los pliegues se solapan, así que el margen de Hoeffding '
                'no se aplica directamente al promedio.</div>', unsafe_allow_html=True)
    st.markdown('<div class="disp" style="font-size:22px;margin:14px 0 4px">Dimensión de la red</div>', unsafe_allow_html=True)
    capas = " → ".join(str(n) for n in d["capas"])
    por_capa = " + ".join(f"{n:,}" for n in d["parametros_por_capa"])
    c4, c5, c6 = st.columns(3)
    c4.markdown(f'<div class="tarjeta"><div class="mut">Parámetros entrenables</div><div class="valor">{d["parametros"]:,}</div>'
                f'<div class="nota">Pesos y sesgos: {por_capa}</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="tarjeta"><div class="mut">Filas de entrenamiento por parámetro</div>'
                f'<div class="valor">{fmt(d["muestras_por_parametro"], 1) if d["muestras_por_parametro"] else "—"}</div>'
                f'<div class="nota">Entrenamiento: {d["muestras_train"]:,} filas con {d["entradas"]} entradas</div></div>',
                unsafe_allow_html=True)
    c6.markdown(f'<div class="tarjeta"><div class="mut">Filas que pediría la regla 10 por parámetro</div>'
                f'<div class="valor">{d["regla_10_por_parametro"]:,}</div>'
                f'<div class="nota">Heurística, no un límite: arquitectura {capas}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="mut" style="font-size:12px;margin-top:6px">La red tiene más parámetros que filas por parámetro '
                'recomendadas. Parte del entrenamiento usa filas duplicadas por el sobremuestreo, así que las filas únicas son '
                'menos que las mostradas.</div>', unsafe_allow_html=True)
    st.markdown('<div class="disp" style="font-size:22px;margin:14px 0 4px">Otras medidas de calidad en prueba</div>',
                unsafe_allow_html=True)
    filas = [
        ("MCC (coeficiente de Matthews)", fmt(x["mcc"]), "1 es perfecto, 0 es azar. No premia predecir la clase mayoritaria."),
        ("Brier (error cuadrático de probabilidades)", fmt(x["brier"]), "0 es perfecto. Mide acierto y confianza juntos."),
        ("ECE (error de calibración)", fmt(x["ece"]), "Diferencia media entre la confianza y el acierto real. 0 es perfecto."),
    ]
    st.markdown('<table class="tabla"><thead><tr><th style="text-align:left">Medida</th><th>Valor</th>'
                '<th style="text-align:left">Cómo leerla</th></tr></thead><tbody>'
                + "".join(f'<tr><td style="text-align:left">{a}</td><td>{b}</td><td style="text-align:left">{c}</td></tr>'
                          for a, b, c in filas) + "</tbody></table>", unsafe_allow_html=True)


@st.dialog("Red neuronal", width="large")
def ficha_dialogo(ds):
    f = cargar_ficha(ds)
    nombre = next((n for d, n in TOP4 if d == ds), ds)
    st.markdown(f'<div class="disp" style="font-size:24px;margin-bottom:4px">{nombre}</div>', unsafe_allow_html=True)
    diagrama, metricas, justificacion = st.tabs(["Diagrama", "Métricas", "Justificación"])
    with diagrama:
        _pestana_diagrama(f)
    with metricas:
        _pestana_metricas(f)
    with justificacion:
        _pestana_justificacion(f)


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
    _bloque_pregunta(ficha_red.PREGUNTA_CARTA.get(ds, {}))
    if st.button("Ver red neuronal", key=f"ver_red_{ds}"):
        ficha_dialogo(ds)
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
    res["cascada"] = None
    if cascada.disponible(ds):
        x = res["nn"]
        if cascada.usa_garantia(ds):
            x = garantia.agregar(ds, [x], [limpio.get(garantia.CAMPO) or 0.0])[0]
        res["cascada"] = cascada.evaluar(ds, [x])[0]
        if cascada.usa_garantia(ds):
            # The ensemble sees Valor_Garantiza, so its class and probabilities replace the single network's.
            res["clase"] = res["clases"][res["cascada"]["clase_idx"]]
            res["probs"] = res["cascada"]["probs"]
    st.session_state[f"res_{ds}"] = res
    hist = st.session_state.setdefault(f"hist_{ds}", [])
    hist.insert(0, {"hora": datetime.now().strftime("%H:%M:%S"), "clase": res["clase"],
                    "confianza": f"{max(res['probs']) * 100:.1f} %".replace(".", ","),
                    "zona_duda": "sí" if zona else "no" if zona is not None else "—"})
    if res["cascada"]:
        hist[0]["estado"] = res["cascada"]["estado"] + (f" ({res['cascada']['motivo']})" if res["cascada"]["motivo"] else "")
    st.session_state[f"paso_{ds}"] = 0
    st.rerun()


def campo_widget(ds, c, datos):
    """Control del campo. SI/NO para booleanos, campo numérico con límite de dígitos y signo, texto para categorías."""
    k, et, previo = clave(ds, c["nombre"]), etiqueta(c["nombre"]), datos.get(c["nombre"])
    cb = dict(on_change=_al_cambiar, args=(ds, c))
    if es_booleano(c):
        opciones = ["SI", "NO"]
        return st.selectbox(et, options=opciones, index=opciones.index(previo) if previo in opciones else None,
                            placeholder="Seleccione SI o NO", key=k, **cb)
    if c["tipo"] == "num":
        minimo, tope = limites_numero(c)
        valor = previo if isinstance(previo, (int, float)) else None
        if c["entero"]:
            return st.number_input(et, min_value=int(minimo), max_value=int(tope), value=valor, step=1,
                                   format="%d", placeholder=f"Hasta {tope}", key=k, **cb)
        return st.number_input(et, min_value=float(minimo), max_value=float(tope), value=valor, step=0.01,
                               format="%.2f", placeholder=f"Hasta {tope}", key=k, **cb)
    return st.text_input(et, value=previo if isinstance(previo, str) else "", key=k,
                         placeholder=marcador(c), **cb)


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
                campo_widget(ds, c, datos)
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
                        f'<td>{h["confianza"]}</td><td>{h["zona_duda"]}</td><td>{h.get("estado", "-")}</td></tr>' for h in hist)
        st.markdown('<table class="tabla"><thead><tr><th>Hora</th><th style="text-align:left">Clase predicha</th><th>Confianza</th>'
                    '<th>Zona de duda</th><th>Cascada</th></tr></thead><tbody>' + filas + "</tbody></table>", unsafe_allow_html=True)

    nn = res.get("nn") if res else None
    st.markdown('<div class="disp" style="font-size:24px;margin:26px 0 4px">Frontera de decisión (plano PCA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="mut" style="font-size:13px;margin-bottom:8px">Cada color es una clase. Los puntos son casos de '
                'validación en su clase real. Las líneas oscuras son el borde entre zonas de un modelo de dos entradas '
                '(las dos primeras componentes PCA), una aproximación a la red. Si clasificó un dato, aparece como estrella '
                'en su posición exacta en ese plano.</div>', unsafe_allow_html=True)
    st.image(frontera.figura(ds, nn=nn, pred=res.get("clase") if nn else None))
    st.markdown('<div class="mut" style="font-size:12px;margin-top:6px">Dos componentes explican solo una parte de la varianza: una zona '
                'que parece cruzar una frontera puede no cruzarla en todas las variables.</div>', unsafe_allow_html=True)


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
