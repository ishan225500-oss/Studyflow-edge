"""
text_extract.py
Turns an uploaded file into plain text as cheaply as possible:

1. .txt / .md          -> read directly
2. PDF with a text layer -> PyMuPDF page.get_text()   (fast, exact, no AI needed)
3. Scanned PDF pages / images -> OCR
     - "easyocr"     : printed notes (default)
     - "handwriting" : EasyOCR finds text lines, TrOCR (handwritten) reads each line

Only pages without usable embedded text are OCR'd, which is the single
biggest latency win for typical lecture PDFs and slides.
"""
import io
from pathlib import Path
from typing import Callable, List, Optional

from config import SETTINGS
import paths
from utils import clean_text, get_logger, timed

log = get_logger("text_extract")

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"}
MIN_TEXT_LAYER_CHARS = 40   # fewer embedded chars than this => treat page as scanned

ProgressFn = Optional[Callable[[str], None]]


class TextExtractor:
    def __init__(self, backend: Optional[str] = None, languages: Optional[List[str]] = None):
        self.backend = backend or SETTINGS.ocr_backend
        self.languages = languages or SETTINGS.ocr_languages
        self._reader = None
        self._trocr = None  # (processor, model)
        self.stats = {"pages": 0, "text_layer_pages": 0, "ocr_pages": 0}

    # ---------- model loading ----------
    def _easyocr(self):
        if self._reader is None:
            import easyocr
            store = paths.models_dir() / "easyocr"
            store.mkdir(parents=True, exist_ok=True)
            has_weights = any(store.glob("*.pth"))
            log.info(f"Loading EasyOCR {self.languages} (weights in {store})")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=False,
                model_storage_directory=str(store),
                download_enabled=not has_weights,
                verbose=False,
            )
        return self._reader

    def _trocr_models(self):
        if self._trocr is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            log.info(f"Loading TrOCR: {SETTINGS.trocr_model}")
            proc = TrOCRProcessor.from_pretrained(SETTINGS.trocr_model)
            model = VisionEncoderDecoderModel.from_pretrained(SETTINGS.trocr_model).eval()
            self._trocr = (proc, model)
        return self._trocr

    def warmup(self):
        self._easyocr()
        if self.backend == "handwriting":
            self._trocr_models()

    # ---------- OCR ----------
    def _ocr_printed(self, image) -> str:
        import numpy as np
        arr = np.array(image.convert("RGB"))
        return "\n".join(self._easyocr().readtext(arr, detail=0, paragraph=True))

    def _ocr_handwriting(self, image) -> str:
        """EasyOCR detects line boxes; TrOCR (a single-line recognizer) reads each."""
        import numpy as np
        import torch

        rgb = image.convert("RGB")
        horizontal, _free = self._easyocr().detect(np.array(rgb), width_ths=0.9, ycenter_ths=0.6)
        boxes = horizontal[0] if horizontal else []
        if not boxes:
            return ""
        # box = [x_min, x_max, y_min, y_max]; read top-to-bottom, left-to-right
        boxes = sorted(boxes, key=lambda b: (round(b[2] / 20), b[0]))
        proc, model = self._trocr_models()
        crops = []
        for x0, x1, y0, y1 in boxes:
            x0, y0 = max(0, int(x0) - 4), max(0, int(y0) - 4)
            x1, y1 = min(rgb.width, int(x1) + 4), min(rgb.height, int(y1) + 4)
            if x1 - x0 > 8 and y1 - y0 > 8:
                crops.append(rgb.crop((x0, y0, x1, y1)))

        lines: List[str] = []
        with torch.no_grad():
            for i in range(0, len(crops), 8):
                pixel_values = proc(images=crops[i:i + 8], return_tensors="pt").pixel_values
                ids = model.generate(pixel_values, max_new_tokens=64)
                lines.extend(proc.batch_decode(ids, skip_special_tokens=True))
        return "\n".join(l.strip() for l in lines if l.strip())

    def _ocr(self, image) -> str:
        if self.backend == "handwriting":
            return self._ocr_handwriting(image)
        return self._ocr_printed(image)

    # ---------- public API ----------
    @timed
    def extract(self, file_path: str, progress: ProgressFn = None) -> str:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {file_path}")
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Use PDF, PNG, JPG or TXT."
            )
        self.stats = {"pages": 0, "text_layer_pages": 0, "ocr_pages": 0}

        if ext in {".txt", ".md"}:
            self.stats["pages"] = 1
            return self._checked(path.read_text(encoding="utf-8", errors="replace"))

        if ext == ".pdf":
            return self._checked(self._extract_pdf(path, progress))

        from PIL import Image, ImageOps
        self.stats["pages"] = self.stats["ocr_pages"] = 1
        if progress:
            progress("Reading text from the image")
        try:
            img = Image.open(path)
        except Image.DecompressionBombError:
            raise ValueError("This image is too large to read. Resize it or take a normal phone photo.")
        with img:
            return self._checked(self._ocr(limit_pixels(ImageOps.exif_transpose(img))))

    @staticmethod
    def _checked(text: str) -> str:
        text = clean_text(text)
        if len(text) > SETTINGS.max_text_chars:
            raise ValueError(
                f"These notes are very long ({len(text):,} characters). "
                "Split them into smaller files, for example one chapter each."
            )
        return text

    def _extract_pdf(self, path: Path, progress: ProgressFn) -> str:
        import fitz  # PyMuPDF
        from PIL import Image

        texts: List[str] = []
        with fitz.open(str(path)) as doc:
            n = doc.page_count
            if n > SETTINGS.max_pdf_pages:
                raise ValueError(
                    f"This PDF has {n} pages; StudyFlow reads up to {SETTINGS.max_pdf_pages} at a time. "
                    "Split it into smaller files, for example one chapter each."
                )
            self.stats["pages"] = n
            # Reading the text layer is cheap, so check every page first and
            # refuse before any slow OCR if too many pages are scanned.
            embedded = [page.get_text("text") or "" for page in doc]
            scanned = [i for i, t in enumerate(embedded) if len(t.strip()) < MIN_TEXT_LAYER_CHARS]
            if len(scanned) > SETTINGS.max_ocr_pages:
                raise ValueError(
                    f"This PDF has {len(scanned)} scanned pages; StudyFlow reads up to "
                    f"{SETTINGS.max_ocr_pages} scanned pages at a time. Split it into smaller files."
                )
            scanned_set = set(scanned)
            done = 0
            for i, text in enumerate(embedded):
                if i not in scanned_set:
                    self.stats["text_layer_pages"] += 1
                    texts.append(text)
                    continue
                done += 1
                if progress:
                    progress(f"Scanning page {i + 1} of {n} ({done} of {len(scanned)} scanned pages)")
                page = doc[i]
                pix = page.get_pixmap(dpi=render_dpi(page.rect.width, page.rect.height))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                self.stats["ocr_pages"] += 1
                texts.append(self._ocr(img))
        return "\n\n".join(t.strip() for t in texts if t.strip())


def max_pixels() -> int:
    return int(SETTINGS.max_image_mp * 1_000_000)


def render_dpi(width_pt: float, height_pt: float, preferred: int = 200) -> int:
    """DPI for rendering a PDF page: 200 for normal pages, lower for huge
    pages (posters, scanned A0 sheets) so the image stays under the pixel cap."""
    area_in2 = max(1e-6, (width_pt / 72.0) * (height_pt / 72.0))
    cap = int((max_pixels() / area_in2) ** 0.5)
    return max(36, min(preferred, cap))


def limit_pixels(img):
    """Downscale images above the pixel cap (large phone photos stay readable
    at 20 MP, and OCR is much faster)."""
    w, h = img.size
    limit = max_pixels()
    if w * h <= limit:
        return img
    scale = (limit / (w * h)) ** 0.5
    from PIL import Image
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python backend/text_extract.py <file.pdf|image|txt>")
        sys.exit(1)
    ex = TextExtractor()
    print(ex.extract(sys.argv[1]))
    print(ex.stats, file=sys.stderr)
