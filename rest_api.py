import os
import warnings
import numpy as np
from PIL import Image
import io
import base64
warnings.filterwarnings('ignore')

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

from inference.classifier import return_classifier
from inference.segmenter import return_segmenter
from segmantation_base.inference import InferenceConfig
from config import Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(['html', 'xml']),
    cache_size=0
)

MAX_FILE_SIZE = 10 * 1024 * 1024


def render_template(request: Request, template_name: str, context: dict, status_code: int = 200):
    context['request'] = request
    template = env.get_template(template_name)
    html_content = template.render(**context)
    return HTMLResponse(content=html_content, status_code=status_code)


config = Config.from_json('config.json')

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
    raise RuntimeError("Classifier initialization failed") from e

segmenter = return_segmenter(InferenceConfig.from_json("segmantation_base//config.json"))

app = FastAPI(title="Image Processing API")
changed_url = "/{process}/ports/{ports}/"


def image_to_base64(image: Image.Image, format: str) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')


def _segmentation_to_pil(seg_result) -> Image.Image:
    if isinstance(seg_result, Image.Image):
        return seg_result.convert("RGB")

    arr = np.asarray(seg_result)

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Unsupported segmentation output shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[2] == 1:
        return Image.fromarray(arr[..., 0], mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(np.ascontiguousarray(arr), mode="RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(np.ascontiguousarray(arr[..., :3]), mode="RGB")
    raise ValueError(f"Unsupported segmentation output shape: {arr.shape}")


def get_base_url(request: Request):
    return str(request.url).rstrip('/')


@app.get(changed_url, response_class=HTMLResponse)
async def read_root(request: Request):
    base_url = get_base_url(request)
    context = {"base_url": base_url}
    return render_template(request, "upload.html", context)


@app.post(changed_url + "upload/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    base_url = get_base_url(request)[:-7]

    try:
        image_data = await file.read()
        if len(image_data) == 0:
            context = {"error": "Загружен пустой файл", "base_url": base_url}
            return render_template(request, "upload.html", context, status_code=400)
        if len(image_data) > MAX_FILE_SIZE:
            context = {"error": f"Файл превышает максимальный размер {MAX_FILE_SIZE // (1024*1024)} МБ", "base_url": base_url}
            return render_template(request, "upload.html", context, status_code=413)
    except Exception as e:
        context = {"error": "Ошибка чтения файла", "base_url": base_url}
        return render_template(request, "upload.html", context, status_code=400)

    if not file.content_type.startswith('image/'):
        context = {"error": "Файл должен быть изображением", "base_url": base_url}
        return render_template(request, "upload.html", context, status_code=400)

    try:
        image = Image.open(io.BytesIO(image_data))
        image.load()
    except Exception as e:
        context = {"error": "Некорректный формат изображения или файл повреждён", "base_url": base_url}
        return render_template(request, "upload.html", context, status_code=400)

    try:
        original_format = image.format or "JPEG"
        original_base64 = image_to_base64(image, original_format)

        probs, labels = classifier(image)
        predictions = []
        colors = getattr(config, 'class_colors',
                        ['#FF4136', '#2ECC40', '#0074D9', '#FF851B'])
        for name, prob, label, color in zip(class_names, probs, labels, colors):
            predictions.append({
                'name': name,
                'prob': float(prob),
                'label': bool(label),
                'color': color
            })

        seg_arr = segmenter(image)
        seg_image = _segmentation_to_pil(seg_arr)
        segmentation_base64 = image_to_base64(seg_image, "PNG")

        mime_type = f"image/{original_format.lower()}"

        context = {
            "original_image": f"data:{mime_type};base64,{original_base64}",
            "segmentation_image": f"data:image/png;base64,{segmentation_base64}",
            "predictions": predictions,
            "message": "Изображение успешно обработано",
            "base_url": base_url,
            "filename": file.filename
        }
        return render_template(request, "result.html", context)

    except Exception as e:
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
        raise HTTPException(status_code=500, detail="Ошибка при скачивании файла")