import os
import logging
import numpy as np
from PIL import Image
import io
import base64
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from torch.cuda import is_available

from inference.classifier import return_classifier
from inference.segmenter import return_segmenter
from config import Config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rest_api")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(['html', 'xml']),
    cache_size=0
)

# Константы
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def render_template(request: Request, template_name: str, context: dict, status_code: int = 200):
    """Рендерит HTML-шаблон с возможностью указать HTTP-статус."""
    context['request'] = request
    template = env.get_template(template_name)
    html_content = template.render(**context)
    return HTMLResponse(content=html_content, status_code=status_code)

config = Config.from_json('config.json')

# Получаем классификатор
classifier_config = {
    'device': config.device,
    'num_classes': config.num_classes,
    'in_height': config.in_height,
    'in_width': config.in_width,
    'model_name': config.model_name,
    'model_mode': config.model_mode,
    'h_load_path': config.h_load_path,
    'transform_mean': config.transform_mean,
    'transform_std': config.transform_std,
    'threshold': config.threshold,
    'class_names': config.class_names,
    'class_colors': config.class_colors,
}

try:
    classifier, class_names = return_classifier(classifier_config)
except Exception as e:
    logger.critical(f"Failed to initialize classifier: {e}", exc_info=True)
    raise RuntimeError("Classifier initialization failed") from e

segmenter = return_segmenter()

app = FastAPI(title="Image Processing API")
changed_url = "/{process}/ports/{ports}/"

def image_to_base64(image: Image.Image, format: str) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

def get_base_url(request: Request):
    return str(request.url).rstrip('/')

@app.get(changed_url, response_class=HTMLResponse)
async def read_root(request: Request):
    base_url = get_base_url(request)
    context = {"base_url": base_url}
    return render_template(request, "upload.html", context)

@app.post(changed_url + "upload/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    base_url = get_base_url(request)[:-7]  # убирается "/upload/"

    # 1. Проверка размера файла
    try:
        image_data = await file.read()
        if len(image_data) == 0:
            logger.warning("Empty file uploaded")
            context = {"error": "Загружен пустой файл", "base_url": base_url}
            return render_template(request, "upload.html", context, status_code=400)
        if len(image_data) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {len(image_data)} bytes")
            context = {"error": f"Файл превышает максимальный размер {MAX_FILE_SIZE // (1024*1024)} МБ", "base_url": base_url}
            return render_template(request, "upload.html", context, status_code=413)
    except Exception as e:
        logger.error(f"Error reading file: {e}", exc_info=True)
        context = {"error": "Ошибка чтения файла", "base_url": base_url}
        return render_template(request, "upload.html", context, status_code=400)

    # 2. Проверка MIME-типа (не обязательная, но дополнительная)
    if not file.content_type.startswith('image/'):
        logger.warning(f"Invalid content type: {file.content_type}")
        context = {"error": "Файл должен быть изображением", "base_url": base_url}
        return render_template(request, "upload.html", context, status_code=400)

    # 3. Попытка открыть изображение через PIL (проверка целостности)
    try:
        image = Image.open(io.BytesIO(image_data))
        # Принудительно загружаем данные, чтобы убедиться, что файл не повреждён
        image.load()
    except Exception as e:
        logger.warning(f"Invalid image file: {e}")
        context = {"error": "Некорректный формат изображения или файл повреждён", "base_url": base_url}
        return render_template(request, "upload.html", context, status_code=400)

    # 4. Основная обработка
    try:
        original_format = image.format or "JPEG"
        original_base64 = image_to_base64(image, original_format)

        probs, labels = classifier(image)
        predictions = []
        # Используем цвета из конфига, если они переданы
        colors = getattr(config, 'class_colors', ['#FF4136', '#2ECC40', '#0074D9', '#FF851B'])
        for name, prob, label, color in zip(class_names, probs, labels, colors):
            predictions.append({
                'name': name,
                'prob': float(prob),
                'label': bool(label),
                'color': color
            })

        seg_arr = segmenter(image)
        seg_image = Image.fromarray((seg_arr * 255).astype(np.uint8)).convert('RGB')
        segmentation_base64 = image_to_base64(seg_image, original_format)

        mime_type = f"image/{original_format.lower()}"

        context = {
            "original_image": f"data:{mime_type};base64,{original_base64}",
            "segmentation_image": f"data:{mime_type};base64,{segmentation_base64}",
            "predictions": predictions,
            "message": "Изображение успешно обработано",
            "base_url": base_url,
            "filename": file.filename
        }
        return render_template(request, "result.html", context)

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        context = {"error": f"Ошибка обработки: {str(e)}", "base_url": base_url}
        return render_template(request, "upload.html", context, status_code=500)

@app.post(changed_url + "download/")
async def download_file(request: Request):
    try:
        form_data = await request.form()
        image_data = form_data["image_data"]
        filename = form_data["filename"]

        header, base64_str = image_data.split(",", 1)
        image_bytes = base64.b64decode(base64_str)
        content_type = header.split(";")[0].split(":")[1]

        return Response(
            content=image_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename=segmentation_{filename}"}
        )
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при скачивании файла")