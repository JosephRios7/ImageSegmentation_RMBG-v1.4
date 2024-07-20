import streamlit as st
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from skimage import io as skio
import os
from datetime import datetime

# Configuración del dispositivo para la inferencia (GPU si está disponible, de lo contrario CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Cargar el modelo de segmentación de imágenes preentrenado
model = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-1.4", trust_remote_code=True)
# Mover el modelo al dispositivo configurado
model.to(device)

# Función para preprocesar la imagen antes de la segmentación


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:  # Si la imagen tiene menos de 3 dimensiones (es en escala de grises), agregar un canal adicional
        im = im[:, :, np.newaxis]
    # Convertir la imagen a un tensor y permutar las dimensiones a (C, H, W)
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(
        im_tensor, 0), size=model_input_size, mode='bilinear')  # Redimensionar la imagen
    # Normalizar los valores de la imagen a [0, 1]
    image = torch.divide(im_tensor, 255.0)
    # Aplicar normalización específica
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image

# Función para postprocesar la imagen después de la segmentación


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    # Redimensionar la imagen al tamaño original
    result = torch.squeeze(F.interpolate(
        result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)  # Obtener el valor máximo de los resultados
    mi = torch.min(result)  # Obtener el valor mínimo de los resultados
    result = (result - mi) / (ma - mi)  # Normalizar los resultados a [0, 1]
    # Convertir el tensor a un arreglo numpy y escalar a [0, 255]
    im_array = (result * 255).permute(1, 2,
                                      0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)  # Eliminar cualquier dimensión adicional
    return im_array

# Función principal para procesar la imagen


def process_image(image_path: str):
    # Leer la imagen usando skio, manejar BytesIO
    orig_im = skio.imread(image_path)
    # Obtener el tamaño original de la imagen
    orig_im_size = orig_im.shape[0:2]

    # Tamaño de entrada del modelo, ajustar según sea necesario
    model_input_size = (256, 256)
    image = preprocess_image(orig_im, model_input_size).to(
        device)  # Preprocesar la imagen y moverla al dispositivo

    with torch.no_grad():  # Desactivar el cálculo de gradientes para la inferencia
        result = model(image)  # Realizar la inferencia con el modelo

    # Postprocesar la imagen segmentada
    result_image = postprocess_image(result[0][0], orig_im_size)

    # Convertir la imagen a escala de grises de un solo canal
    pil_im = Image.fromarray(result_image).convert("L")
    # Crear una nueva imagen RGBA con fondo transparente
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    # Pegar la imagen original usando la segmentada como máscara
    no_bg_image.paste(Image.fromarray(orig_im), mask=pil_im)

    return orig_im, no_bg_image

# Función para descargar una imagen desde una URL


def download_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        st.error("Error al descargar la imagen")
        return None

# Función para guardar las imágenes en la galería


def save_images(original, segmented):
    if not os.path.exists("gallery"):
        os.makedirs("gallery")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    original_path = f"gallery/{timestamp}_original.png"
    segmented_path = f"gallery/{timestamp}_segmented.png"
    original.save(original_path)
    segmented.save(segmented_path)
    return original_path, segmented_path

# Función para mostrar la galería de imágenes


def show_gallery():
    if not os.path.exists("gallery"):
        st.write("No hay imágenes en la galería.")
        return

    images = sorted(os.listdir("gallery"))
    image_pairs = [(images[i], images[i+1]) for i in range(0, len(images), 2)]

    for original, segmented in image_pairs:
        st.image(os.path.join("gallery", original),
                 caption="Imagen Original", use_column_width=True)
        st.image(os.path.join("gallery", segmented),
                 caption="Imagen Segmentada", use_column_width=True)


# Interfaz de Streamlit
st.set_page_config(page_title="Segmentación de Imágenes", layout="wide")
st.title("Segmentación de Imágenes con RMBG v1.4")

# Barra de navegación
nav = st.sidebar.radio("Navegación", ["Subir Imagen", "Ver Galería"])

if nav == "Subir Imagen":
    # Selección de método para cargar la imagen
    option = st.selectbox("Selecciona el método para cargar la imagen:",
                          ("Subir desde el PC", "Ingresar URL"))

    if option == "Subir desde el PC":
        uploaded_file = st.file_uploader(
            "Elige una imagen...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    elif option == "Ingresar URL":
        url = st.text_input("Ingresa la URL de la imagen:")
        if url:
            image = download_image_from_url(url)

    if "image" in locals():
        st.image(image, caption="Imagen Original", use_column_width=True)
        image = image.convert("RGB")
        image_path = "input.jpg"
        image.save(image_path, "JPEG")

        # Procesar la imagen
        orig_im, result_image = process_image(image_path)

        # Mostrar la imagen segmentada
        st.image(result_image, caption="Imagen Segmentada",
                 use_column_width=True)

        # Guardar las imágenes
        orig_path, seg_path = save_images(
            Image.fromarray(orig_im), result_image)
        st.success(f"Imágenes guardadas en: {orig_path} y {seg_path}")

elif nav == "Ver Galería":
    show_gallery()
