
import streamlit as st

st.set_page_config(
    page_title="CrutchForce",
    page_icon="🩼",
    layout="wide"
)

st.title("🩼 CrutchForce")

st.subheader(
    "Sistema de análisis biomecánico de la marcha con muletas"
)

st.info(
    "Aplicación académica para analizar videos de personas "
    "que utilizan muletas."
)

st.header("1. Datos de la persona")

col1, col2 = st.columns(2)

with col1:
    peso = st.number_input(
        "Peso corporal (kg)",
        min_value=20.0,
        max_value=200.0,
        value=70.0
    )

with col2:
    altura = st.number_input(
        "Altura (m)",
        min_value=1.0,
        max_value=2.3,
        value=1.70
    )

st.header("2. Cargar video")

video = st.file_uploader(
    "Selecciona el video de una persona utilizando muletas",
    type=["mp4", "mov", "avi"]
)

if video is not None:

    st.success("Video cargado correctamente")

    st.video(video)

    if st.button("Analizar video", type="primary"):

        st.info(
            "El módulo de análisis biomecánico "
            "se integrará en la siguiente versión."
        )

st.divider()

st.caption(
    "CrutchForce | Proyecto académico de Ingeniería Biomédica"
)
