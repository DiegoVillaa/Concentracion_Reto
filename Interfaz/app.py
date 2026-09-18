"""Interfaz web para explorar señales y utilizar el clasificador REHAB."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_utils import (
    CHANNEL_NAMES,
    combine_sensor_groups,
    extract_features,
    feature_names,
    load_uploaded_npy,
    project_root,
    select_repetition,
    split_into_windows,
)


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "modelo_random_forest.joblib"
RAW_DATA_DIR = project_root() / "Rehab_exercise" / "d02_processed_data"

CHANNEL_COLORS = {
    "pitch1": "#7C5CFC", "yaw1": "#2DD4BF", "roll1": "#F59E0B",
    "pitch2": "#EC4899", "yaw2": "#38BDF8", "roll2": "#A3E635",
    "f1": "#FB7185", "f2": "#C084FC", "f3": "#22D3EE",
    "f4": "#FBBF24", "f5": "#4ADE80", "pitch3": "#818CF8",
}


st.set_page_config(
    page_title="REHAB Motion Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
      :root { --violet:#7867ff; --cyan:#23d5d0; --ink:#eef1ff; --muted:#9aa4c3; }
      .stApp { background: radial-gradient(circle at 12% 5%, #212457 0, #101329 34%, #080b18 76%); }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
      h1,h2,h3 { font-family:'Space Grotesk',sans-serif!important; letter-spacing:-.03em; }
      [data-testid="stSidebar"] { background:linear-gradient(180deg,#121630,#090c1b); border-right:1px solid #292e51; }
      .hero { padding:28px 32px; border:1px solid rgba(140,128,255,.25); border-radius:24px;
              background:linear-gradient(120deg,rgba(124,92,252,.23),rgba(35,213,208,.08)); margin-bottom:22px; }
      .eyebrow { color:#7de8df; font-size:.78rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; }
      .hero h1 { color:white; font-size:3rem; margin:.25rem 0 .5rem; }
      .hero p { color:#b8c0d9; font-size:1.08rem; max-width:780px; margin:0; }
      .metric-card { min-height:138px; padding:20px; border-radius:20px; border:1px solid #292e51;
                     background:linear-gradient(145deg,rgba(30,35,73,.92),rgba(15,18,40,.92)); }
      .metric-label { color:#8f9ab9; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; }
      .metric-value { color:white; font:700 2rem 'Space Grotesk'; margin:.45rem 0 .15rem; }
      .metric-detail { color:#aab3cc; font-size:.85rem; }
      .window-card { padding:18px; border-radius:18px; border:1px solid #30365c; background:#12162c; text-align:center; }
      .window-number { color:#8f9ab9; font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; }
      .window-label { color:white; font:700 1.7rem 'Space Grotesk'; margin:.25rem 0; }
      .confidence { color:#74e3d9; font-size:.88rem; }
      .note { padding:14px 17px; color:#b8c0d9; background:rgba(35,213,208,.07); border-left:3px solid #23d5d0; border-radius:10px; }
      div[data-testid="stFileUploader"] { border:1px dashed #4b5388; border-radius:16px; padding:8px; }
      .stButton>button { border-radius:12px; border:0; color:white; font-weight:700;
                         background:linear-gradient(90deg,#7867ff,#4e9cff); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def metric_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def load_demo_signal(activity: str, repetition: int) -> np.ndarray:
    group_1 = np.load(RAW_DATA_DIR / f"{activity}_1.npy", mmap_mode="r", allow_pickle=False)
    group_2 = np.load(RAW_DATA_DIR / f"{activity}_2.npy", mmap_mode="r", allow_pickle=False)
    return combine_sensor_groups(group_1[repetition], group_2[repetition])


def available_activities() -> list[str]:
    return [f"{number:03d}" for number in range(16) if (RAW_DATA_DIR / f"{number:03d}_1.npy").exists()]


artifact = load_artifact()
if artifact is None:
    st.error("No se encontró el modelo entrenado. Ejecuta `python entrenar_modelo.py` dentro de la carpeta Interfaz.")
    st.stop()

model = artifact["model"]
metadata = artifact["metadata"]

with st.sidebar:
    st.markdown("## 🧠 REHAB Lab")
    st.caption("Clasificación de movimientos con sensores portátiles")
    st.divider()
    input_mode = st.radio("Fuente de la señal", ["Ejemplo del dataset", "Cargar archivos .npy"])

    expected_activity = None
    signal = None
    if input_mode == "Ejemplo del dataset":
        activities = available_activities()
        activity = st.selectbox("Actividad conocida", activities, index=0)
        group_shape = np.load(RAW_DATA_DIR / f"{activity}_1.npy", mmap_mode="r", allow_pickle=False).shape
        repetition = st.slider("Número de repetición", 0, group_shape[0] - 1, 0)
        expected_activity = activity
        signal = load_demo_signal(activity, repetition)
        st.success(f"Ejemplo listo: actividad {activity}, repetición {repetition}")
    else:
        st.caption("Carga los dos grupos que corresponden a la misma actividad y repetición.")
        upload_1 = st.file_uploader("Grupo de sensores 1", type=["npy"], key="group1")
        upload_2 = st.file_uploader("Grupo de sensores 2", type=["npy"], key="group2")
        if upload_1 and upload_2:
            try:
                array_1 = load_uploaded_npy(upload_1)
                array_2 = load_uploaded_npy(upload_2)
                count_1 = 1 if array_1.ndim == 2 else array_1.shape[0]
                count_2 = 1 if array_2.ndim == 2 else array_2.shape[0]
                if count_1 != count_2:
                    raise ValueError("Los archivos no contienen la misma cantidad de repeticiones.")
                repetition = st.number_input("Repetición a analizar", 0, count_1 - 1, 0)
                signal = combine_sensor_groups(
                    select_repetition(array_1, int(repetition)),
                    select_repetition(array_2, int(repetition)),
                )
            except Exception as error:
                st.error(str(error))

    st.divider()
    st.caption("Modelo definitivo · Random Forest · 16 actividades")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Rehabilitation intelligence</div>
      <h1>REHAB Motion Lab</h1>
      <p>Explora una señal cinemática y observa cómo el modelo identifica patrones de movimiento en cada ventana temporal.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if signal is None:
    st.info("Selecciona un ejemplo o carga los dos archivos de sensores para comenzar.")
    st.stop()

try:
    windows = split_into_windows(signal)
    features = extract_features(windows)
    feature_frame = pd.DataFrame(features, columns=feature_names())
    predictions = model.predict(feature_frame)
    probabilities = model.predict_proba(feature_frame)
except Exception as error:
    st.error(f"No fue posible analizar la señal: {error}")
    st.stop()

confidence = probabilities.max(axis=1)
dominant_activity = pd.Series(predictions).mode().iloc[0]
average_confidence = confidence.mean()

metric_columns = st.columns(4)
with metric_columns[0]:
    metric_card("Entrada validada", "880 × 12", "Puntos temporales × canales")
with metric_columns[1]:
    metric_card("Ventanas", "4", "220 puntos, sin traslape")
with metric_columns[2]:
    metric_card("Actividad dominante", str(dominant_activity), "Resumen visual de las 4 predicciones")
with metric_columns[3]:
    metric_card("Confianza promedio", f"{average_confidence:.1%}", "Probabilidad máxima media")

st.markdown("## Resultado por ventana")
window_columns = st.columns(4)
for index, column in enumerate(window_columns):
    with column:
        match = expected_activity is not None and predictions[index] == expected_activity
        status = " · coincide" if match else ""
        st.markdown(
            f'<div class="window-card"><div class="window-number">Ventana {index + 1} · puntos {index * 220 + 1}–{(index + 1) * 220}</div>'
            f'<div class="window-label">{predictions[index]}</div>'
            f'<div class="confidence">{confidence[index]:.1%} de confianza{status}</div></div>',
            unsafe_allow_html=True,
        )

if expected_activity is not None:
    correct = int(np.sum(predictions == expected_activity))
    st.markdown(
        f'<div class="note">La etiqueta conocida de este ejemplo es <strong>{expected_activity}</strong>. '
        f'El modelo acertó <strong>{correct} de 4 ventanas</strong>.</div>',
        unsafe_allow_html=True,
    )

st.caption("Cada ventana es una observación independiente, igual que en el entrenamiento y la evaluación del proyecto. La actividad dominante se muestra únicamente como resumen visual; no reemplaza las predicciones por ventana.")

left, right = st.columns([1.5, 1])
with left:
    st.markdown("## Señal cinemática")
    selected_channels = st.multiselect(
        "Canales visibles",
        CHANNEL_NAMES,
        default=["pitch1", "yaw1", "roll1"],
        max_selections=6,
    )
    if selected_channels:
        signal_frame = pd.DataFrame(signal, columns=CHANNEL_NAMES).reset_index(names="punto")
        long_signal = signal_frame.melt("punto", value_vars=selected_channels, var_name="canal", value_name="valor")
        figure = px.line(
            long_signal,
            x="punto",
            y="valor",
            color="canal",
            color_discrete_map=CHANNEL_COLORS,
            template="plotly_dark",
        )
        for boundary in [220, 440, 660]:
            figure.add_vline(x=boundary, line_dash="dot", line_color="#667099", opacity=0.75)
        figure.update_layout(
            height=420, margin=dict(l=15, r=15, t=25, b=10), legend_title_text="Canal",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,13,30,.65)",
            xaxis_title="Punto temporal", yaxis_title="Valor normalizado",
        )
        st.plotly_chart(figure, use_container_width=True)

with right:
    st.markdown("## Confianza detallada")
    selected_window = st.selectbox("Ventana", [1, 2, 3, 4]) - 1
    probability_frame = pd.DataFrame(
        {"actividad": model.classes_, "probabilidad": probabilities[selected_window]}
    ).sort_values("probabilidad", ascending=True).tail(6)
    bars = px.bar(
        probability_frame,
        x="probabilidad",
        y="actividad",
        orientation="h",
        color="probabilidad",
        color_continuous_scale=["#31385e", "#7867ff", "#23d5d0"],
        template="plotly_dark",
    )
    bars.update_layout(
        height=420, margin=dict(l=10, r=10, t=25, b=10), coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,13,30,.65)",
        xaxis_tickformat=".0%", xaxis_title="Probabilidad", yaxis_title="Actividad",
    )
    st.plotly_chart(bars, use_container_width=True)

st.markdown("## Mapa de probabilidades")
heatmap = go.Figure(
    data=go.Heatmap(
        z=probabilities,
        x=model.classes_,
        y=["Ventana 1", "Ventana 2", "Ventana 3", "Ventana 4"],
        colorscale=[[0, "#10142b"], [0.4, "#5446aa"], [1, "#24d5cf"]],
        colorbar=dict(title="Prob."),
        hovertemplate="%{y}<br>Actividad %{x}<br>Probabilidad %{z:.1%}<extra></extra>",
    )
)
heatmap.update_layout(
    height=330, margin=dict(l=15, r=15, t=20, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,13,30,.65)",
    xaxis_title="Actividad REHAB", yaxis_title="",
)
st.plotly_chart(heatmap, use_container_width=True)

with st.expander("¿Cómo se obtiene la predicción?"):
    st.markdown(
        f"""
        1. Se unen los dos grupos de sensores para obtener **880 puntos y 12 canales**.
        2. La señal se divide en **cuatro ventanas consecutivas de 220 puntos**.
        3. En cada canal se calculan media, mediana, desviación, mínimo, máximo, rango,
           percentiles 25 y 75, rango intercuartílico y RMS: **120 características por ventana**.
        4. El Random Forest definitivo clasifica cada ventana entre las actividades **000–015**.

        El modelo fue ajustado con **{metadata['training_rows']:,} ventanas** de train + validation.
        En el test reservado obtuvo **{metadata['test_accuracy']:.2%} de exactitud** y
        **{metadata['test_f1_macro']:.4f} de F1 macro**.
        """
    )
