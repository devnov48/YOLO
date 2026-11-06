# Detector de objetos YOLO

Aplicación interactiva en Streamlit para detectar objetos en imágenes JPG/PNG y
videos MP4 usando Ultralytics YOLO.

## Requisitos

- Python 3.10 o posterior
- Dependencias de `requirements.txt`

## Instalación y ejecución

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

Abre la dirección local que muestra Streamlit, normalmente
`http://localhost:8501`.

## Modelos

La app incluye `yolo26n-2.pt` y también ofrece los modelos YOLOv8 Nano, Small y
Medium. Ultralytics descargará los pesos YOLOv8 la primera vez que se
seleccionen. Para usar el modelo incluido, conserva `yolo26n-2.pt` en la misma
carpeta que `app.py`.

## Recuento de video

El recuento suma las detecciones a través de todos los fotogramas del video; no
realiza seguimiento de objetos únicos.
