
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import urllib.request
import subprocess
import imageio_ffmpeg

# ============================================
# CONFIGURACION DE LA PAGINA
# ============================================

st.set_page_config(
    page_title="CrutchForce",
    page_icon="🩼",
    layout="wide"
)

st.title("🩼 CrutchForce - analizar la marcha de personas que utilizan muletas, evaluando sus movimientos y estimando la fuerza aplicada sobre ellas, para identificar posibles alteraciones biomecánicas y contribuir a una movilidad más segura.")
st.subheader")
    "Sistema de analisis biomecanico de marcha con muletas"
)

st.info(
    "Proyecto academico de Ingenieria Biomedica. "
    "Analisis de movimiento mediante video."
)

# ============================================
# MODELO MEDIAPIPE
# ============================================

@st.cache_resource
def obtener_modelo():

    url = (
        "https://storage.googleapis.com/"
        "mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/"
        "pose_landmarker_lite.task"
    )

    ruta = os.path.join(
        tempfile.gettempdir(),
        "crutchforce_pose.task"
    )

    if not os.path.exists(ruta) or os.path.getsize(ruta) == 0:
        with urllib.request.urlopen(url, timeout=60) as respuesta:
            with open(ruta, "wb") as archivo:
                archivo.write(respuesta.read())

    return ruta


# ============================================
# CALCULO DE ANGULOS
# ============================================

def calcular_angulo(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    v1 = a - b
    v2 = c - b

    denominador = (
        np.linalg.norm(v1) *
        np.linalg.norm(v2)
    )

    if denominador < 1e-8:
        return np.nan

    coseno = np.dot(v1, v2) / denominador
    coseno = np.clip(coseno, -1, 1)

    return float(np.degrees(np.arccos(coseno)))


# ============================================
# CONEXIONES DEL ESQUELETO
# ============================================

CONEXIONES = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]


# ============================================
# PROCESAMIENTO DEL VIDEO
# ============================================

def analizar_video(video_bytes):

    modelo = obtener_modelo()

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    Opciones = mp.tasks.vision.PoseLandmarkerOptions
    Modo = mp.tasks.vision.RunningMode

    configuracion = Opciones(
        base_options=BaseOptions(
            model_asset_path=modelo
        ),
        running_mode=Modo.VIDEO,
        num_poses=1
    )

    registros = []
    detectados = 0

    with tempfile.TemporaryDirectory() as carpeta:

        entrada = os.path.join(carpeta, "entrada.mp4")
        temporal = os.path.join(carpeta, "temporal.mp4")
        salida = os.path.join(carpeta, "resultado.mp4")

        with open(entrada, "wb") as archivo:
            archivo.write(video_bytes)

        captura = cv2.VideoCapture(entrada)

        if not captura.isOpened():
            raise ValueError(
                "No se pudo abrir el video. "
                "Intenta utilizar un archivo MP4."
            )

        fps = captura.get(cv2.CAP_PROP_FPS)

        if fps <= 0 or not np.isfinite(fps):
            captura.release()
            raise ValueError(
                "No se pudo determinar la velocidad del video."
            )

        ancho_original = int(
            captura.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        alto_original = int(
            captura.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if ancho_original <= 0 or alto_original <= 0:
            captura.release()
            raise ValueError("Dimensiones invalidas del video.")

        escala = min(
            1.0,
            720 / max(ancho_original, alto_original)
        )

        ancho = max(2, int(ancho_original * escala))
        alto = max(2, int(alto_original * escala))

        ancho -= ancho % 2
        alto -= alto % 2

        escritor = cv2.VideoWriter(
            temporal,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (ancho, alto)
        )

        if not escritor.isOpened():
            captura.release()
            raise RuntimeError(
                "No fue posible crear el video procesado."
            )

        barra = st.progress(0, text="Procesando video...")

        total = int(
            captura.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Limite para la primera version
        max_cuadros = min(
            900,
            max(1, int(fps * 30))
        )

        if total > 0:
            max_cuadros = min(max_cuadros, total)

        numero = 0

        try:
            with PoseLandmarker.create_from_options(
                configuracion
            ) as detector:

                while numero < max_cuadros:

                    correcto, frame = captura.read()

                    if not correcto:
                        break

                    frame = cv2.resize(
                        frame, (ancho, alto)
                    )

                    rgb = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )

                    imagen = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb)
                    )

                    tiempo_ms = int(
                        numero * 1000 / fps
                    )

                    resultado = detector.detect_for_video(
                        imagen,
                        tiempo_ms
                    )

                    tiempo = numero / fps

                    fila = {
                        "Tiempo (s)": tiempo,
                        "Codo izquierdo (grados)": np.nan,
                        "Codo derecho (grados)": np.nan,
                        "Hombro izquierdo (grados)": np.nan,
                        "Hombro derecho (grados)": np.nan,
                        "Muneca izquierda Y": np.nan,
                        "Muneca derecha Y": np.nan
                    }

                    if resultado.pose_landmarks:

                        puntos = resultado.pose_landmarks[0]

                        # Coordenadas normalizadas
                        xy = [
                            (p.x, p.y) for p in puntos
                        ]

                        # Coordenadas en pixeles
                        pixel = [
                            (
                                int(p.x * ancho),
                                int(p.y * alto)
                            )
                            for p in puntos
                        ]

                        # Solo utilizar articulaciones visibles
                        def visible(indices):
                            return all(
                                puntos[i].visibility >= 0.5
                                for i in indices
                            )

                        # Codo izquierdo
                        if visible([11, 13, 15]):
                            fila["Codo izquierdo (grados)"] = (
                                calcular_angulo(
                                    xy[11], xy[13], xy[15]
                                )
                            )

                        # Codo derecho
                        if visible([12, 14, 16]):
                            fila["Codo derecho (grados)"] = (
                                calcular_angulo(
                                    xy[12], xy[14], xy[16]
                                )
                            )

                        # Hombro izquierdo
                        if visible([13, 11, 23]):
                            fila["Hombro izquierdo (grados)"] = (
                                calcular_angulo(
                                    xy[13], xy[11], xy[23]
                                )
                            )

                        # Hombro derecho
                        if visible([14, 12, 24]):
                            fila["Hombro derecho (grados)"] = (
                                calcular_angulo(
                                    xy[14], xy[12], xy[24]
                                )
                            )

                        if visible([15]):
                            fila["Muneca izquierda Y"] = puntos[15].y

                        if visible([16]):
                            fila["Muneca derecha Y"] = puntos[16].y

                        # Dibujar conexiones
                        for a, b in CONEXIONES:

                            if visible([a, b]):
                                cv2.line(
                                    frame,
                                    pixel[a],
                                    pixel[b],
                                    (0, 255, 0),
                                    2
                                )

                        # Dibujar articulaciones
                        for i in [
                            11, 12, 13, 14, 15, 16,
                            23, 24, 25, 26, 27, 28
                        ]:
                            if visible([i]):
                                cv2.circle(
                                    frame,
                                    pixel[i],
                                    5,
                                    (0, 0, 255),
                                    -1
                                )

                        detectados += 1

                    cv2.putText(
                        frame,
                        f"Tiempo: {tiempo:.2f} s",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    registros.append(fila)
                    escritor.write(frame)

                    numero += 1

                    if numero % 10 == 0:
                        barra.progress(
                            min(numero / max_cuadros, 1.0),
                            text="Analizando articulaciones..."
                        )

        finally:
            captura.release()
            escritor.release()
            barra.empty()

        if numero == 0:
            raise ValueError(
                "No se pudieron procesar cuadros del video."
            )

        # Convertir video para reproducir en navegador
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel", "error",
                "-i", temporal,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                salida
            ],
            check=True,
            timeout=180
        )

        with open(salida, "rb") as archivo:
            video_resultado = archivo.read()

    datos = pd.DataFrame(registros)

    return video_resultado, datos, detectados, numero


# ============================================
# DATOS DE LA PERSONA
# ============================================

st.header("1. Datos de la persona")

col1, col2 = st.columns(2)

with col1:
    peso = st.number_input(
        "Peso corporal (kg)",
        min_value=20.0,
        max_value=200.0,
        value=70.0,
        step=0.5
    )

with col2:
    altura = st.number_input(
        "Altura (m)",
        min_value=1.0,
        max_value=2.3,
        value=1.70,
        step=0.01
    )


# ============================================
# CARGAR VIDEO
# ============================================

st.header("2. Cargar video")

video = st.file_uploader(
    "Selecciona un video MP4",
    type=["mp4"]
)

if video is not None:

    st.success("Video cargado correctamente")
    st.video(video)

    if st.button(
        "Analizar video",
        type="primary",
        use_container_width=True
    ):

        try:
            with st.spinner(
                "Realizando analisis biomecanico..."
            ):

                procesado, datos, detectados, total = (
                    analizar_video(video.getvalue())
                )

            st.session_state["resultado"] = {
                "video": procesado,
                "datos": datos,
                "detectados": detectados,
                "total": total,
                "nombre": video.name
            }

        except Exception as error:
            st.error(
                "No se pudo procesar el video. "
                f"Detalle: {error}"
            )

    # ========================================
    # RESULTADOS
    # ========================================

    if "resultado" in st.session_state:

        r = st.session_state["resultado"]

        if r["nombre"] == video.name:

            datos = r["datos"]

            st.header("3. Resultados biomecanicos")

            porcentaje = (
                100 * r["detectados"] / r["total"]
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Cuadros analizados",
                r["total"]
            )

            c2.metric(
                "Deteccion de postura",
                f"{porcentaje:.1f} %"
            )

            c3.metric(
                "Peso ingresado",
                f"{peso:.1f} kg"
            )

            st.subheader("Video con articulaciones")

            st.video(r["video"])

            st.download_button(
                "Descargar video procesado",
                data=r["video"],
                file_name="CrutchForce_analizado.mp4",
                mime="video/mp4"
            )

            if datos.notna().sum().sum() > 0:

                st.subheader(
                    "Ángulo de flexión y extensión de ambos codos (el angulo de los codos durante la marcha)"
                )

                columnas_codo = [
                    "Codo izquierdo (grados)",
                    "Codo derecho (grados)"
                ]

                st.line_chart(
                    datos.set_index("Tiempo (s)")[
                        columnas_codo
                    ]
                )

                st.subheader(
                    "Variacion angular de hombro. (los angulos de los hombros durante el desplazamiento)"
                )

                columnas_hombro = [
                    "Hombro izquierdo (grados)",
                    "Hombro derecho (grados)"
                ]

                st.line_chart(
                    datos.set_index("Tiempo (s)")[
                        columnas_hombro
                    ]
                )

                st.subheader(
                    "Posición de muñecas. (las oscilaciones que presentan las manos al utilizar las muletas) "
                )

                st.line_chart(
                    datos.set_index("Tiempo (s)")[
                        [
                            "Muneca izquierda Y",
                            "Muneca derecha Y"
                        ]
                    ]
                )

                st.subheader("Datos cuadro por cuadro")

                st.dataframe(datos)

                st.download_button(
                    "Descargar resultados CSV",
                    data=datos.to_csv(index=False).encode("utf-8-sig"),
                    file_name="CrutchForce_resultados.csv",
                    mime="text/csv"
                )

            st.warning(
                "Esta version analiza postura y movimiento. "
                "No mide fuerzas ni confirma apoyos sobre el suelo. "
                "Los angulos se calculan a partir de la imagen 2D."
            )

st.divider()

st.caption(
    "CrutchForce 2.0 | Proyecto academico "
    "de Ingenieria Biomedica"
)
