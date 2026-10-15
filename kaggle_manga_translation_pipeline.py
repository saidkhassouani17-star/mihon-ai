# %% [markdown]
# محرك ترجمة المانجا: OCR + LaMa + Gemini + رسم عربي
#
# انسخ هذه الخلايا إلى Kaggle Notebook، أو شغّل الملف مباشرة في محرر يدعم صيغة `# %%`.

# %% [markdown]
# 1) Environment Setup
#
# شغّل Paddle على CPU افتراضيًا لتجنب حجز الذاكرة مع LaMa. يمكن تجربة `PADDLE_DEVICE=gpu`
# فقط إذا كان إصدار Paddle CUDA مطابقًا لبيئة Kaggle.

# %%
import os
import sys
import subprocess

PADDLE_DEVICE = os.environ.get("PADDLE_DEVICE", "cpu").lower()
os.environ["PADDLE_DEVICE"] = PADDLE_DEVICE

packages = [
    "paddleocr>=2.8.1",
    "simple-lama-inpainting",
    "opencv-python-headless",
    "pillow>=10.0.0",
    "arabic-reshaper",
    "python-bidi",
    "google-genai>=1.0.0",
    "fastapi",
    "uvicorn",
    "python-multipart",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

# CPU Paddle يتجنب تعارض CUDA مع PyTorch/LaMa. استخدم نسخة Kaggle CUDA المطابقة فقط عند الحاجة.
if PADDLE_DEVICE == "cpu":
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "paddlepaddle>=3.0.0",
    ])
else:
    # Paddle يوفر حزم GPU مرتبطة بإصدار CUDA؛ ثبّت الإصدار المناسب لبيئة Kaggle لديك.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "paddlepaddle-gpu>=3.0.0",
    ])

# خط عربي واضح. الرابط يشير إلى ملف TTF خام ثابت في مستودع Google Fonts.
FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/cairo/static/"
    "Cairo-Bold.ttf"
)
FONT_PATH = "/kaggle/working/Cairo-Bold.ttf"
if not os.path.exists(FONT_PATH):
    subprocess.check_call(["wget", "-q", "-O", FONT_PATH, FONT_URL])

# %% [markdown]
# 2) Imports and shared helpers

# %%
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from arabic_reshaper import reshape
from bidi.algorithm import get_display


Polygon = List[Tuple[float, float]]
Box = List[int]


def _as_polygon(points: Any) -> Polygon:
    """Normalize Paddle's polygon formats to [(x, y), ...]."""
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in array]


def _polygon_box(polygon: Polygon) -> Box:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [int(math.floor(min(xs))), int(math.floor(min(ys))),
            int(math.ceil(max(xs))), int(math.ceil(max(ys)))]


def _box_gap(first: Box, second: Box) -> Tuple[float, float]:
    """Return horizontal and vertical gaps; overlap counts as zero gap."""
    horizontal = max(0.0, max(first[0], second[0]) - min(first[2], second[2]))
    vertical = max(0.0, max(first[1], second[1]) - min(first[3], second[3]))
    return horizontal, vertical


# %% [markdown]
# 3) OCR and bubble grouping

# %%
@dataclass
class MergedText:
    text: str
    bbox: Box
    raw_polygons: List[Polygon] = field(default_factory=list)
    language: str = "en"


class MangaOCRProcessor:
    """Detect text lines with PaddleOCR and group nearby lines into bubbles."""

    RTL_GROUPING_LANGUAGES = {"japan", "ch"}

    def __init__(self, languages: Sequence[str] = ("japan", "korean", "ch", "en"),
                 use_gpu: bool = False):
        from paddleocr import PaddleOCR

        self.languages = tuple(languages)
        self.ocr_engines = {}
        for language in self.languages:
            kwargs = {"lang": language, "use_doc_orientation_classify": False,
                      "use_doc_unwarping": False, "use_textline_orientation": False}
            # Older PaddleOCR releases use `use_angle_cls` instead of the newer flags.
            try:
                self.ocr_engines[language] = PaddleOCR(
                    device="gpu:0" if use_gpu else "cpu", **kwargs
                )
            except TypeError:
                self.ocr_engines[language] = PaddleOCR(
                    use_angle_cls=False,
                    use_gpu=use_gpu,
                    lang=language,
                )

    @staticmethod
    def _extract_lines(result: Any) -> Iterable[Tuple[Polygon, str, float]]:
        """Read both PaddleOCR 2.x and 3.x result shapes."""
        if result is None:
            return
        if hasattr(result, "json"):
            result = result.json
        if isinstance(result, dict):
            result = result.get("res", result)
            polygons = result.get("dt_polys", result.get("polys", []))
            texts = result.get("rec_texts", result.get("texts", []))
            scores = result.get("rec_scores", result.get("scores", []))
            scores = list(scores) if scores is not None else []
            for polygon, text, score in zip(polygons, texts, scores or [1.0] * len(texts)):
                yield _as_polygon(polygon), str(text).strip(), float(score)
            return
        for page in result if isinstance(result, list) else [result]:
            if isinstance(page, (list, tuple)) and len(page) == 2:
                polygon, recognition = page
                text, score = recognition if isinstance(recognition, (list, tuple)) else (recognition, 1.0)
                yield _as_polygon(polygon), str(text).strip(), float(score)
                continue
            if isinstance(page, dict):
                yield from MangaOCRProcessor._extract_lines(page)
                continue
            lines = getattr(page, "rec_texts", None)
            polygons = getattr(page, "dt_polys", None)
            scores = getattr(page, "rec_scores", None)
            if lines is not None and polygons is not None:
                scores = list(scores) if scores is not None else []
                for polygon, text, score in zip(polygons, lines, scores or [1.0] * len(lines)):
                    yield _as_polygon(polygon), str(text).strip(), float(score)

    def detect(self, image: Image.Image | np.ndarray, min_score: float = 0.45,
               source_lang: Optional[str] = None) -> List[MergedText]:
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        detected: List[MergedText] = []
        engines = self.ocr_engines.items() if source_lang is None else ((source_lang, self.ocr_engines[source_lang]),)
        for language, engine in engines:
            output = engine.predict(image) if hasattr(engine, "predict") else engine.ocr(image, cls=False)
            for result in output:
                for polygon, text, score in self._extract_lines(result):
                    if text and score >= min_score:
                        detected.append(MergedText(text, _polygon_box(polygon), [polygon], language))
        return self.merge_text_boxes(detected)

    @staticmethod
    def merge_text_boxes(ocr_results: Sequence[MergedText | Dict[str, Any]],
                          x_thresh: float = 40.0, y_thresh: float = 28.0) -> List[MergedText]:
        """Merge connected nearby lines while preserving every original polygon.

        Two regions are connected when their horizontal and vertical gaps are both within
        threshold, or when one region overlaps the other on both axes. Connected components
        are then ordered right-to-left for Japanese/Chinese and left-to-right otherwise.
        """
        items: List[MergedText] = []
        for item in ocr_results:
            if isinstance(item, MergedText):
                items.append(item)
            else:
                polygon = item.get("raw_polygon", item.get("polygon", item.get("raw_polygons")))
                polygon = _as_polygon(polygon[0] if polygon and np.asarray(polygon).ndim == 3 else polygon)
                items.append(MergedText(str(item["text"]), _polygon_box(polygon), [polygon], item.get("language", "en")))
        groups: List[List[MergedText]] = []
        unused = set(range(len(items)))
        while unused:
            seed = unused.pop()
            group = [seed]
            changed = True
            while changed:
                changed = False
                for index in tuple(unused):
                    if any(
                        _box_gap(items[index].bbox, items[member].bbox)[0] <= x_thresh
                        and _box_gap(items[index].bbox, items[member].bbox)[1] <= y_thresh
                        for member in group
                    ):
                        unused.remove(index)
                        group.append(index)
                        changed = True
            groups.append([items[index] for index in group])

        merged: List[MergedText] = []
        for group in groups:
            language = group[0].language
            rtl = language in MangaOCRProcessor.RTL_GROUPING_LANGUAGES
            group.sort(key=lambda item: (item.bbox[1], item.bbox[0]), reverse=rtl)
            left = min(item.bbox[0] for item in group)
            top = min(item.bbox[1] for item in group)
            right = max(item.bbox[2] for item in group)
            bottom = max(item.bbox[3] for item in group)
            separator = " "
            merged.append(MergedText(
                text=separator.join(item.text for item in group),
                bbox=[left, top, right, bottom],
                raw_polygons=[polygon for item in group for polygon in item.raw_polygons],
                language=language,
            ))
        merged.sort(key=lambda item: (item.bbox[1], -item.bbox[0] if item.language in MangaOCRProcessor.RTL_GROUPING_LANGUAGES else item.bbox[0]))
        return merged


# %% [markdown]
# 4) Precise masking and LaMa inpainting

# %%
class LaMaInpainter:
    def __init__(self, device: Optional[str] = None):
        from simple_lama_inpainting import SimpleLama
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        try:
            self.model = SimpleLama(device=self.device)
        except TypeError:
            self.model = SimpleLama()

    @staticmethod
    def create_precise_mask(image_shape: Sequence[int], raw_polygons: Sequence[Polygon],
                            dilation_kernel_size: int = 9) -> Image.Image:
        height, width = int(image_shape[0]), int(image_shape[1])
        mask = np.zeros((height, width), dtype=np.uint8)
        polygons = [np.rint(np.asarray(polygon)).astype(np.int32) for polygon in raw_polygons]
        if polygons:
            cv2.fillPoly(mask, polygons, 255)
        kernel_size = max(1, int(dilation_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
        return Image.fromarray(mask, mode="L")

    def inpaint(self, image: Image.Image | np.ndarray, raw_polygons: Sequence[Polygon],
                dilation_kernel_size: int = 9) -> Image.Image:
        source = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
        mask = self.create_precise_mask(source.size[::-1], raw_polygons, dilation_kernel_size)
        cleaned = self.model(source, mask)
        return cleaned.convert("RGB") if isinstance(cleaned, Image.Image) else Image.fromarray(np.asarray(cleaned)).convert("RGB")


# %% [markdown]
# 5) One-request contextual Gemini translation

# %%
class GeminiBatchTranslator:
    SYSTEM_INSTRUCTION = (
        "أنت مترجم محترف للمانجا والمانهوا. ترجم الحوارات إلى عربية طبيعية ودرامية، "
        "وحافظ على المعنى والشخصيات والسياق ونبرة المشهد. لا تشرح ولا تضف نصًا. "
        "أعد كائن JSON صحيحًا فقط، بنفس المفاتيح الرقمية التي أرسلها المستخدم، "
        "وقيمة كل مفتاح ترجمة عربية نصية."
    )

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        from google import genai
        self.model = model
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    def translate(self, texts: Sequence[MergedText | str]) -> Dict[str, str]:
        source = {str(index): (item.text if isinstance(item, MergedText) else str(item))
                  for index, item in enumerate(texts, start=1)}
        if not source:
            return {}
        prompt = "ترجم القيم التالية مع إبقاء المفاتيح كما هي:\n" + json.dumps(source, ensure_ascii=False)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "system_instruction": self.SYSTEM_INSTRUCTION,
                    "response_mime_type": "application/json",
                    "temperature": 0.35,
                },
            )
            parsed = json.loads(response.text)
            return {key: str(parsed.get(key, source[key])) for key in source}
        except Exception as error:
            print(f"Gemini translation failed; using original text: {error}")
            return source


# %% [markdown]
# 6) Arabic typesetting and rendering

# %%
class ArabicTypesetter:
    def __init__(self, font_path: str = FONT_PATH, margin_ratio: float = 0.10):
        self.font_path = font_path
        self.margin_ratio = margin_ratio

    @staticmethod
    def _shape(text: str) -> str:
        return get_display(reshape(text))

    def _wrap(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
              max_width: int) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            shaped = self._shape(candidate)
            if current and draw.textbbox((0, 0), shaped, font=font, stroke_width=3)[2] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
        return lines

    def draw_text(self, cleaned_image: Image.Image, bbox: Box, arabic_text: str,
                  initial_font_size: Optional[int] = None) -> Image.Image:
        image = cleaned_image.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        x_min, y_min, x_max, y_max = bbox
        box_width, box_height = max(1, x_max - x_min), max(1, y_max - y_min)
        safe_width = max(1, int(box_width * (1.0 - self.margin_ratio)))
        safe_height = max(1, int(box_height * (1.0 - self.margin_ratio)))
        font_size = initial_font_size or max(12, int(box_height * 0.45))
        chosen_lines: List[str] = []
        chosen_font: Optional[ImageFont.FreeTypeFont] = None
        while font_size >= 8:
            font = ImageFont.truetype(self.font_path, font_size)
            lines = self._wrap(draw, arabic_text, font, safe_width)
            line_boxes = [draw.textbbox((0, 0), self._shape(line), font=font, stroke_width=3) for line in lines]
            line_height = max(1, int(font_size * 1.25))
            total_height = line_height * len(lines)
            widest = max((box[2] - box[0] for box in line_boxes), default=0)
            if widest <= safe_width and total_height <= safe_height:
                chosen_lines, chosen_font = lines, font
                break
            font_size -= 1
        if chosen_font is None:
            chosen_font = ImageFont.truetype(self.font_path, 8)
            chosen_lines = self._wrap(draw, arabic_text, chosen_font, safe_width)

        line_height = max(1, int(chosen_font.size * 1.25))
        total_height = line_height * len(chosen_lines)
        cursor_y = y_min + (box_height - total_height) // 2
        for line in chosen_lines:
            display_line = self._shape(line)
            bounds = draw.textbbox((0, 0), display_line, font=chosen_font, stroke_width=3)
            text_width = bounds[2] - bounds[0]
            cursor_x = x_min + (box_width - text_width) // 2 - bounds[0]
            draw.text((cursor_x, cursor_y), display_line, font=chosen_font, fill=(0, 0, 0),
                      stroke_width=3, stroke_fill=(255, 255, 255))
            cursor_y += line_height
        return image


# %% [markdown]
# 7) Example end-to-end page function

# %%
def translate_page(image_path: str, output_path: str, api_key: Optional[str] = None) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    ocr = MangaOCRProcessor(use_gpu=PADDLE_DEVICE == "gpu")
    detected = ocr.detect(image)
    all_polygons = [polygon for item in detected for polygon in item.raw_polygons]
    cleaned = LaMaInpainter().inpaint(image, all_polygons, dilation_kernel_size=9)
    translations = GeminiBatchTranslator(api_key=api_key).translate(detected)
    typesetter = ArabicTypesetter()
    result = cleaned
    for index, item in enumerate(detected, start=1):
        result = typesetter.draw_text(result, item.bbox, translations.get(str(index), item.text))
    result.save(output_path, format="PNG")
    return result


# %% [markdown]
# مثال Kaggle:
# ```python
# result = translate_page("/kaggle/input/my-manga/page.png", "/kaggle/working/page_ar.png")
# display(result)
# ```

# %% [markdown]
# 8) Singleton pipeline, MD5 cache, and FastAPI server

# %%
import asyncio
import hashlib
import io
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response


class MangaTranslationEngine:
    """One process-wide owner for OCR, LaMa, typesetting, and translated image cache."""

    _instance: Optional["MangaTranslationEngine"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "MangaTranslationEngine":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialize(*args, **kwargs)
                    cls._instance = instance
        return cls._instance

    def _initialize(self, cache_dir: str = "/kaggle/working/manga_cache",
                    memory_cache_size: int = 32) -> None:
        self._process_lock = threading.Lock()
        self._memory_cache: OrderedDict[str, bytes] = OrderedDict()
        self._memory_cache_size = memory_cache_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        use_gpu = PADDLE_DEVICE == "gpu"
        # These three objects are created once, so repeated requests reuse their state/VRAM.
        self.ocr = MangaOCRProcessor(use_gpu=use_gpu)
        self.inpainter = LaMaInpainter()
        self.typesetter = ArabicTypesetter()

    @staticmethod
    def _cache_key(image_bytes: bytes, source_lang: str) -> str:
        digest = hashlib.md5(image_bytes, usedforsecurity=False).hexdigest()
        return f"{digest}_{source_lang.lower()}"

    def _get_cached(self, key: str) -> Optional[bytes]:
        cached = self._memory_cache.get(key)
        if cached is not None:
            self._memory_cache.move_to_end(key)
            return cached
        cache_path = self.cache_dir / f"{key}.webp"
        if cache_path.is_file():
            cached = cache_path.read_bytes()
            self._memory_cache[key] = cached
            self._memory_cache.move_to_end(key)
            while len(self._memory_cache) > self._memory_cache_size:
                self._memory_cache.popitem(last=False)
            return cached
        return None

    def _put_cached(self, key: str, content: bytes) -> None:
        self._memory_cache[key] = content
        self._memory_cache.move_to_end(key)
        while len(self._memory_cache) > self._memory_cache_size:
            self._memory_cache.popitem(last=False)
        cache_path = self.cache_dir / f"{key}.webp"
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_bytes(content)
        temporary_path.replace(cache_path)

    def process(self, image_bytes: bytes, source_lang: str,
                api_key: Optional[str] = None) -> bytes:
        source_lang = source_lang.lower().strip()
        if source_lang not in self.ocr.ocr_engines:
            supported = ", ".join(self.ocr.ocr_engines)
            raise ValueError(f"Unsupported source_lang '{source_lang}'. Use: {supported}")
        key = self._cache_key(image_bytes, source_lang)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        # Protect non-thread-safe native inference objects when concurrent requests arrive.
        with self._process_lock:
            cached = self._get_cached(key)
            if cached is not None:
                return cached
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            detected = self.ocr.detect(image, source_lang=source_lang)
            polygons = [polygon for item in detected for polygon in item.raw_polygons]
            cleaned = self.inpainter.inpaint(image, polygons, dilation_kernel_size=9)
            translations = GeminiBatchTranslator(api_key=api_key).translate(detected)
            result = cleaned
            for index, item in enumerate(detected, start=1):
                result = self.typesetter.draw_text(
                    result, item.bbox, translations.get(str(index), item.text)
                )
            output = io.BytesIO()
            result.save(output, format="WEBP", quality=85, method=4)
            content = output.getvalue()
            self._put_cached(key, content)
            return content


app = FastAPI(title="Manga Translation Server", version="1.0.0")
engine: Optional[MangaTranslationEngine] = None


@app.on_event("startup")
def load_models_once() -> None:
    global engine
    engine = MangaTranslationEngine()


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        import torch
        gpu_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        vram = round(torch.cuda.memory_allocated(0) / 1024**3, 3) if gpu_available else 0.0
    except (ImportError, RuntimeError):
        gpu_available, gpu_name, vram = False, None, 0.0
    return {
        "status": "ok",
        "gpu": {"available": gpu_available, "name": gpu_name, "allocated_gb": vram},
        "paddle_device": PADDLE_DEVICE,
        "models_loaded": engine is not None,
    }


@app.post("/translate-page")
async def translate_page_endpoint(
    image: UploadFile = File(...),
    source_lang: str = Form("japan"),
    api_key: Optional[str] = Form(None),
) -> Response:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="The uploaded file must be an image")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    if engine is None:
        raise HTTPException(status_code=503, detail="Models are still loading")
    try:
        webp_bytes = await asyncio.to_thread(engine.process, image_bytes, source_lang, api_key)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (OSError, UnidentifiedImageError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid image: {error}") from error
    return Response(content=webp_bytes, media_type="image/webp")


# %% [markdown]
# 9) Start Uvicorn in the background

# %%
import time
import uvicorn


def start_uvicorn(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn-manga-server")
    thread.start()
    # Give the socket a short, bounded startup window without blocking forever.
    for _ in range(50):
        if server.started:
            print(f"FastAPI listening on http://{host}:{port}")
            return thread
        time.sleep(0.1)
    raise RuntimeError("Uvicorn did not start within five seconds")


# %% [markdown]
# 10) Download cloudflared, create a Quick Tunnel, and print its public URL

# %%
import platform
import re
import stat
import subprocess
import urllib.request


def start_cloudflared(port: int = 8000, binary_path: str = "/kaggle/working/cloudflared") -> subprocess.Popen:
    binary = Path(binary_path)
    if not binary.exists():
        architecture = platform.machine().lower()
        asset = "cloudflared-linux-amd64" if architecture in {"x86_64", "amd64"} else "cloudflared-linux-arm64"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}"
        urllib.request.urlretrieve(url, binary)
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

    process = subprocess.Popen(
        [str(binary), "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    public_url = None
    deadline = time.time() + 30
    while time.time() < deadline and process.poll() is None:
        line = process.stdout.readline() if process.stdout else ""
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            public_url = match.group(0)
            break
    if not public_url:
        process.terminate()
        raise RuntimeError("cloudflared did not provide a trycloudflare.com URL")
    print("=" * 72)
    print(f"PUBLIC URL: {public_url}")
    print(f"POST {public_url}/translate-page")
    print("=" * 72)
    return process


# %% [markdown]
# تشغيل الخادم والنفق في Kaggle:
# ```python
# server_thread = start_uvicorn(port=8000)
# cloudflared_process = start_cloudflared(port=8000)
# ```
#
# اختبار من الهاتف:
# ```bash
# curl -X POST "PUBLIC_URL/translate-page" \
#   -F "image=@page.jpg" -F "source_lang=japan" \
#   --output translated.webp
# ```
