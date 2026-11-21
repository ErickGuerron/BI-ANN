"""Lienzo movible de la red: todas las neuronas y conexiones, con forma de neurona.

Modo "entrenamiento": color y grosor de cada conexión según el peso de la época actual.
Modo "prueba": color y grosor según la contribución (activación de origen x peso) para un caso.
Modo "final": los pesos del modelo ya entrenado, sin época ni caso (se usa en la ficha de la red).
El HTML se entrega a st.components.v1.html; no depende de librerías externas.
"""
import json

COLOR_NAVY = "#15233A"
COLOR_OCRE = "#9A5B2E"
COLOR_FONDO = "#F4F6F9"
COLOR_TINTA = "#4A5B70"


def html_lienzo(capas, coefs, modo="entrenamiento", entradas=None, clases=None,
                activaciones=None, probas=None, alto=600, umbral=0.0, separacion=240):
    """capas: tamaños por capa; coefs: pesos por capa (n_in x n_out); activaciones: por capa (opcional).

    separacion: distancia horizontal entre capas. Con muchas entradas la red es muy alta; una separación
    mayor la acerca a la proporción del lienzo y deja más espacio útil al ajustar la vista."""
    datos = {
        "separacion": separacion,
        "capas": list(capas),
        "coefs": [w.tolist() if hasattr(w, "tolist") else w for w in coefs],
        "modo": modo,
        "entradas": list(entradas) if entradas is not None else None,
        "clases": list(clases) if clases is not None else None,
        "activaciones": [a.tolist() if hasattr(a, "tolist") else a for a in activaciones] if activaciones is not None else None,
        "probas": [float(p) for p in probas] if probas is not None else None,
        "umbral": umbral,
    }
    payload = json.dumps(datos, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<style>
  html,body{{margin:0;height:100%;background:{COLOR_FONDO};font-family:'Instrument Sans',system-ui,sans-serif;color:{COLOR_NAVY};overflow:hidden}}
  #wrap{{position:relative;width:100%;height:{alto}px;background:{COLOR_FONDO};border:1px solid #B9C4D0;border-radius:2px;box-sizing:border-box}}
  canvas{{width:100%;height:100%;display:block;cursor:grab}}
  canvas.drag{{cursor:grabbing}}
  #leyenda{{position:absolute;left:12px;top:10px;font-size:12px;color:{COLOR_TINTA};background:rgba(244,246,249,.92);padding:8px 10px;border:1px solid #C9D2DC;border-radius:2px;line-height:1.6}}
  #leyenda b{{color:{COLOR_NAVY};font-weight:600}}
  #ctl{{position:absolute;right:12px;top:10px;display:flex;gap:8px;align-items:center;font-size:12px;color:{COLOR_TINTA};background:rgba(244,246,249,.92);padding:6px 10px;border:1px solid #C9D2DC;border-radius:2px}}
  #ctl button{{font:500 12px 'Instrument Sans',sans-serif;background:transparent;color:{COLOR_NAVY};border:1px solid #8C9BAD;border-radius:2px;padding:4px 10px;cursor:pointer}}
  #ctl input{{width:110px}}
  #tip{{position:absolute;pointer-events:none;display:none;background:{COLOR_NAVY};color:{COLOR_FONDO};font-size:12px;padding:6px 9px;border-radius:2px;max-width:280px;line-height:1.45}}
</style></head>
<body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="leyenda"><b>Azul marino</b>: peso positivo<br><b>Ocre</b>: peso negativo<br>Grosor: magnitud<br><span id="modo"></span></div>
  <div id="ctl"><span>Mostrar desde</span><input id="umbral" type="range" min="0" max="0.9" step="0.01"><button id="ajustar">Ajustar vista</button></div>
  <div id="tip"></div>
</div>
<script>
const D = {payload};
const NAVY = "{COLOR_NAVY}", OCRE = "{COLOR_OCRE}", FONDO = "{COLOR_FONDO}", TINTA = "{COLOR_TINTA}";
const wrap = document.getElementById('wrap'), cv = document.getElementById('c'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip'), umbralEl = document.getElementById('umbral');
document.getElementById('modo').textContent = D.modo === 'prueba' ? 'Contribución de cada conexión a este caso' : (D.modo === 'final' ? 'Pesos del modelo final' : 'Pesos de la época actual');
umbralEl.value = D.umbral;
const L = D.capas.length;
const COL_X = D.capas.map((_, l) => 60 + l * D.separacion);
const ESP = D.capas.map((_, l) => (l === 0 ? 14 : 26));
const POS = D.capas.map((n, l) => Array.from({{length: n}}, (_, i) => ({{x: COL_X[l], y: (i - (n - 1) / 2) * ESP[l]}})));
const MAXW = D.coefs.map(W => Math.max(1e-9, ...W.flat().map(Math.abs)));
const ACT = D.activaciones;
let MAXC = [];
if (ACT) {{
  MAXC = D.coefs.map((W, l) => {{
    let m = 1e-9;
    for (let i = 0; i < W.length; i++) for (let j = 0; j < W[i].length; j++) m = Math.max(m, Math.abs(ACT[l][i] * W[i][j]));
    return m;
  }});
}}
const ACT_MAX = ACT ? ACT.map(a => Math.max(1e-9, ...a.map(Math.abs))) : null;
let W = 0, H = 0, dpr = window.devicePixelRatio || 1, scale = 1, ox = 0, oy = 0, arrastrando = null, anclaje = null, movida = false;
const SOMA = 4.2, BOUT = 13, DEND = 12;

function ajustar() {{
  const xs = POS.flat().map(p => p.x), ys = POS.flat().map(p => p.y);
  const x0 = Math.min(...xs) - 40, x1 = Math.max(...xs) + 140, y0 = Math.min(...ys) - 40, y1 = Math.max(...ys) + 40;
  const RESERVA = 190;                    // ancho de la leyenda: la red empieza a su derecha y no queda tapada
  scale = Math.min((W - RESERVA) / (x1 - x0), H / (y1 - y0));
  ox = RESERVA + (W - RESERVA - (x1 - x0) * scale) / 2 - x0 * scale;
  oy = (H - (y1 - y0) * scale) / 2 - y0 * scale;
  dibujar();
}}

function redimensionar() {{
  dpr = window.devicePixelRatio || 1;
  W = wrap.clientWidth; H = wrap.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  dibujar();
}}

function neurona(x, y, relleno, borde) {{
  ctx.lineWidth = 1;
  ctx.strokeStyle = borde;
  ctx.fillStyle = relleno;
  ctx.beginPath();                       // dendritas
  [[-DEND, -5], [-DEND - 1, 0], [-DEND, 5]].forEach(([dx, dy]) => {{
    ctx.moveTo(x - SOMA, y); ctx.quadraticCurveTo(x - SOMA - 4, y + dy / 2, x + dx, y + dy);
  }});
  ctx.stroke();
  ctx.beginPath();                       // axón
  ctx.moveTo(x + SOMA, y); ctx.lineTo(x + BOUT - 2, y);
  ctx.stroke();
  ctx.beginPath(); ctx.arc(x, y, SOMA, 0, Math.PI * 2); ctx.fillStyle = relleno; ctx.fill(); ctx.stroke(); // soma
  ctx.beginPath(); ctx.arc(x + BOUT, y, 2, 0, Math.PI * 2); ctx.fillStyle = borde; ctx.fill(); // terminal sináptica
}}

function dibujar() {{
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = FONDO; ctx.fillRect(0, 0, W, H);
  ctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * ox, dpr * oy);
  const umbral = parseFloat(umbralEl.value) || 0;
  for (let l = 0; l < L - 1; l++) {{
    const Wl = D.coefs[l];
    for (let i = 0; i < Wl.length; i++) {{
      const s = POS[l][i];
      for (let j = 0; j < Wl[i].length; j++) {{
        const w = Wl[i][j];
        let mag, col;
        if (ACT) {{ const c = ACT[l][i] * w; mag = Math.abs(c) / MAXC[l]; col = c >= 0 ? NAVY : OCRE; }}
        else {{ mag = Math.abs(w) / MAXW[l]; col = w >= 0 ? NAVY : OCRE; }}
        if (mag < umbral) continue;
        const d = POS[l + 1][j];
        ctx.globalAlpha = 0.12 + 0.7 * mag;
        ctx.lineWidth = 0.25 + 1.3 * mag;
        ctx.strokeStyle = col;
        ctx.beginPath();
        ctx.moveTo(s.x + BOUT, s.y);
        const cx = (s.x + d.x) / 2;
        ctx.bezierCurveTo(cx, s.y, cx, d.y, d.x - DEND, d.y);
        ctx.stroke();
      }}
    }}
  }}
  ctx.globalAlpha = 1;
  for (let l = 0; l < L; l++) {{
    POS[l].forEach((p, i) => {{
      let relleno = FONDO;
      if (ACT) {{ const a = Math.min(1, Math.abs(ACT[l][i]) / ACT_MAX[l]); relleno = `rgba(21,35,58,${{0.08 + 0.92 * a}})`; }}
      neurona(p.x, p.y, relleno, TINTA);
    }});
    ctx.fillStyle = TINTA; ctx.font = "12px 'Instrument Sans', sans-serif";
    const etiqueta = l === 0 ? `Entradas · ${{D.capas[0]}}` : (l === L - 1 ? `Salida · ${{D.capas[l]}}` : `Oculta ${{l}} · ${{D.capas[l]}}`);
    ctx.fillText(etiqueta, COL_X[l] - 30, POS[l][0].y - 22);
  }}
  if (D.clases && D.probas) {{
    const xs = COL_X[L - 1] + 22;
    D.clases.forEach((c, k) => {{
      const y = POS[L - 1][k] ? POS[L - 1][k].y : 0;
      ctx.fillStyle = NAVY; ctx.font = "600 12px 'Instrument Sans', sans-serif";
      ctx.fillText(`${{c}}  ${{D.probas[k].toFixed(3).replace('.', ',')}}`, xs, y + 4);
    }});
  }}
}}

function mundo(ev) {{
  const r = cv.getBoundingClientRect();
  return {{x: (ev.clientX - r.left - ox) / scale, y: (ev.clientY - r.top - oy) / scale, px: ev.clientX - r.left, py: ev.clientY - r.top}};
}}

cv.addEventListener('pointerdown', ev => {{ movida = true; arrastrando = {{x: ev.clientX, y: ev.clientY, ox, oy}}; cv.classList.add('drag'); cv.setPointerCapture(ev.pointerId); }});
cv.addEventListener('pointerup', () => {{ arrastrando = null; cv.classList.remove('drag'); }});
cv.addEventListener('pointermove', ev => {{
  if (arrastrando) {{
    ox = arrastrando.ox + (ev.clientX - arrastrando.x); oy = arrastrando.oy + (ev.clientY - arrastrando.y);
    dibujar(); tip.style.display = 'none'; return;
  }}
  const m = mundo(ev);
  let mejor = null, dmin = 9 / scale;
  for (let l = 0; l < L; l++) POS[l].forEach((p, i) => {{
    const d = Math.hypot(p.x - m.x, p.y - m.y);
    if (d < dmin) {{ dmin = d; mejor = {{l, i}}; }}
  }});
  if (!mejor) {{ tip.style.display = 'none'; return; }}
  let texto;
  if (mejor.l === 0) texto = `<b>${{D.entradas ? D.entradas[mejor.i] : 'entrada ' + mejor.i}}</b>` + (ACT ? `<br>valor: ${{ACT[0][mejor.i].toFixed(3).replace('.', ',')}}` : '');
  else if (mejor.l === L - 1) texto = `<b>${{D.clases ? D.clases[mejor.i] : 'salida ' + mejor.i}}</b>` + (D.probas ? `<br>probabilidad: ${{D.probas[mejor.i].toFixed(3).replace('.', ',')}}` : '');
  else texto = `<b>Oculta ${{mejor.l}}, neurona ${{mejor.i + 1}}</b>` + (ACT ? `<br>activación: ${{ACT[mejor.l][mejor.i].toFixed(3).replace('.', ',')}}` : '');
  tip.innerHTML = texto; tip.style.display = 'block';
  tip.style.left = Math.min(m.px + 14, W - 240) + 'px'; tip.style.top = (m.py + 14) + 'px';
}});
cv.addEventListener('wheel', ev => {{
  ev.preventDefault(); movida = true;
  const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const f = Math.exp(-ev.deltaY * 0.0015), nueva = Math.min(4, Math.max(0.05, scale * f));
  const k = nueva / scale; ox = mx - (mx - ox) * k; oy = my - (my - oy) * k; scale = nueva;
  dibujar();
}}, {{passive: false}});
umbralEl.addEventListener('input', dibujar);
document.getElementById('ajustar').addEventListener('click', () => {{ movida = false; ajustar(); }});
window.addEventListener('resize', () => {{ redimensionar(); if (!movida) ajustar(); }});
if (window.ResizeObserver) new ResizeObserver(() => {{ if (wrap.clientWidth !== W || wrap.clientHeight !== H) {{ redimensionar(); if (!movida) ajustar(); }} }}).observe(wrap);
redimensionar(); ajustar();
</script>
</body></html>"""
