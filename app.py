"""Aplicación Streamlit para detectar objetos con modelos YOLO."""

import math
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


# Configura el título y el diseño de la página.
st.set_page_config(page_title="Detector de objetos YOLO", page_icon="🎯")
st.title("Detector de objetos con YOLO")
st.write("Carga una imagen JPG/PNG o un video MP4 para detectar objetos.")

# Define los controles de entrada en la barra lateral.
model_options = {
    "YOLOv8 Nano (yolov8n.pt)": "yolov8n.pt",
    "YOLOv8 Small (yolov8s.pt)": "yolov8s.pt",
    "YOLOv8 Medium (yolov8m.pt)": "yolov8m.pt",
    "YOLO26 Nano (yolo26n-2.pt)": str(
        Path(__file__).resolve().parent / "yolo26n-2.pt"
    ),
}
selected_model = st.sidebar.selectbox("Modelo YOLO", list(model_options))
model_name = model_options[selected_model]
conf_threshold = st.sidebar.slider(
    "Umbral de confianza",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
)
uploaded_file = st.sidebar.file_uploader(
    "Carga una imagen o un video",
    type=["jpg", "jpeg", "png", "mp4"],
)


# Mantiene el modelo en memoria entre ejecuciones de Streamlit.
@st.cache_resource(show_spinner="Cargando modelo YOLO...")
def load_model(name: str) -> YOLO:
    return YOLO(name)


def class_name_for(names: dict | list, class_id: int) -> str:
    """Devuelve el nombre legible de la clase detectada."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def update_counts(result, names: dict | list, counts: Counter) -> None:
    """Suma al contador cada detección de un resultado YOLO."""
    if result.boxes is None:
        return
    for class_id_value in result.boxes.cls.tolist():
        counts[class_name_for(names, int(class_id_value))] += 1


def show_counts(counts: Counter, note: str | None = None) -> None:
    """Muestra el total detectado por clase en una tabla."""
    st.subheader("Recuento de detecciones")
    if note:
        st.caption(note)
    if not counts:
        st.info("No se detectaron objetos.")
        return

    st.dataframe(
        [
            {"Clase": name, "Detecciones": total}
            for name, total in counts.most_common()
        ],
        hide_index=True,
        width="stretch",
    )


def process_uploaded_file(uploaded_file) -> None:
    """Procesa el archivo seleccionado por el usuario."""
    model = load_model(model_name)
    counts: Counter = Counter()
    is_video = Path(uploaded_file.name).suffix.lower() == ".mp4"

    if not is_video:
        # Abre la imagen, ejecuta la detección y presenta el resultado anotado.
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            st.error(f"No se pudo abrir la imagen: {error}")
            return

        with st.spinner("Analizando imagen..."):
            result = model.predict(
                source=original_image,
                conf=conf_threshold,
                verbose=False,
            )[0]

        update_counts(result, model.names, counts)
        # result.plot() produce BGR; Streamlit espera una imagen RGB.
        annotated_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

        original_column, result_column = st.columns(2)
        with original_column:
            st.subheader("Imagen original")
            st.image(original_image, width="stretch")
        with result_column:
            st.subheader("Imagen procesada")
            st.image(annotated_image, width="stretch")

        show_counts(counts)
        return

    # Guarda temporalmente el video, procesa sus fotogramas y escribe el resultado.
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.mp4"
        output_path = Path(temp_dir) / "output.mp4"
        input_path.write_bytes(uploaded_file.getvalue())

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            st.error("No se pudo abrir el video. Comprueba que sea un MP4 válido.")
            return

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0

        success, frame = capture.read()
        if not success:
            capture.release()
            st.error("El video no contiene fotogramas legibles.")
            return

        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            st.error("No se pudo crear el video procesado con el códec disponible.")
            return

        progress = st.progress(0.0)
        status = st.empty()
        frame_index = 0

        try:
            with st.spinner("Procesando video..."):
                while success:
                    result = model.predict(
                        source=frame,
                        conf=conf_threshold,
                        verbose=False,
                    )[0]
                    update_counts(result, model.names, counts)
                    writer.write(result.plot())
                    frame_index += 1

                    if total_frames > 0:
                        progress.progress(min(frame_index / total_frames, 1.0))
                    status.write(f"Fotogramas procesados: {frame_index}")
                    success, frame = capture.read()
        finally:
            capture.release()
            writer.release()

        progress.progress(1.0)
        status.write(f"Video procesado: {frame_index} fotogramas.")
        st.subheader("Video procesado")
        st.video(output_path.read_bytes())

        show_counts(
            counts,
            note=(
                "En videos, el recuento suma las detecciones de todos los "
                "fotogramas; no identifica objetos únicos."
            ),
        )


# No depende de st.stop() para detenerse: la comprobación también funciona
# al ejecutar el archivo fuera del contexto normal de Streamlit.
if uploaded_file is None:
    st.info("Selecciona un archivo en la barra lateral para comenzar.")
else:
    process_uploaded_file(uploaded_file)
