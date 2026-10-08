
import streamlit as st

st.set_page_config(
    page_title="CrutchForce",
    page_icon="🩼",
    layout="wide"
)

st.title("🩼 CrutchForce")

st.subheader(
    "Análisis biomecánico de marcha con muletas"
)

st.info(
    "Aplicación académica de Ingeniería Biomédica"
)

st.header("1. Datos de la persona")

peso = st.number_input(
    "Peso corporal (kg)",
    min_value=20.0,
    max_value=200.0,
    value=70.0
)

altura = st.number_input(
    "Altura (m)",
    min_value=1.0,
    max_value=2.3,
    value=1.70
)

st.header("2. Cargar video")

video = st.file_uploader(
    "Selecciona un video",
    type=["mp4", "mov", "avi"]
)

if video is not None:
    st.success("Video cargado correctamente")
    st.video(video)

    if st.button("Analizar video", type="primary"):
        st.info(
            "El módulo de análisis biomecánico "
            "se integrará próximamente."
        )

st.divider()

st.caption(
    "CrutchForce | Proyecto de Ingeniería Biomédica"
)
