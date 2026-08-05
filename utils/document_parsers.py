import io
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps

import utils.patch_torchvision  # noqa: F401

logger = logging.getLogger(__name__)

try:
    import docx
except ImportError:
    docx = None

try:
    import pytesseract

    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    tess_exe = shutil.which("tesseract") or next((p for p in tesseract_paths if os.path.exists(p)), None)
    if tess_exe:
        pytesseract.pytesseract.tesseract_cmd = tess_exe
except ImportError:
    pytesseract = None

try:
    from rapidocr_onnxruntime import RapidOCR
    _rapidocr_engine = None

    def get_rapidocr_engine():
        global _rapidocr_engine
        if _rapidocr_engine is None:
            _rapidocr_engine = RapidOCR()
        return _rapidocr_engine

    # Suppress verbose RapidOCR info logs unless the app explicitly enables debug.
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    logging.getLogger("rapidocr").setLevel(logging.WARNING)
except Exception:
    RapidOCR = None

try:
    import easyocr
    _easyocr_reader = None

    def get_easyocr_reader():
        global _easyocr_reader
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(["en"], gpu=False)
        return _easyocr_reader
except Exception:
    easyocr = None

try:
    from docling.document_converter import DocumentConverter
    _docling_converter = None

    def get_docling_converter():
        global _docling_converter
        if _docling_converter is None:
            _docling_converter = DocumentConverter()
        return _docling_converter
except Exception:
    DocumentConverter = None

# Configurable OCR Engine Order and Boosts
OCR_ENGINE_ORDER = os.getenv("OCR_ENGINE_ORDER", "rapidocr,docling,easyocr,tesseract").split(",")
ENGINE_BOOST = {
    "rapidocr": 1.25,
    "docling": 1.20,
    "easyocr": 1.10,
    "tesseract": 1.0,
}

NOISE_PATTERNS = [
    r"^\s*(page|pg|p\.?)\s*\d+(\s*(?:/|of)\s*\d+)?\s*$",
    r"^\s*(copyright|all rights reserved|confidential|draft|version)\b.*$",
    r"^\s*(table of contents|contents|index|back|next|previous|home)\s*$",
]


def is_noise_line(line: str) -> bool:
    normalized = line.strip().lower()
    if not normalized:
        return True
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, normalized):
            return True
    return False


def remove_noise_from_text(text: str) -> str:
    lines = [line for line in text.splitlines() if not is_noise_line(line)]
    return "\n".join(lines).strip()


OCR_DPI = int(os.getenv("OCR_DPI", "220"))
MIN_OCR_TEXT_LENGTH = int(os.getenv("MIN_OCR_TEXT_LENGTH", "8"))


def normalize_ocr_text(text: str) -> str:
    lines = []
    for line in str(text or "").splitlines():
        # 1. Insert space between glued field names & colons: e.g. FieldName:12345 -> FieldName : 12345
        line = re.sub(r"([a-zA-Z0-9])(:)([a-zA-Z0-9])", r"\1 \2 \3", line)
        line = re.sub(r"([a-zA-Z0-9])(:)\b", r"\1 \2 ", line)

        # 2. Universal CamelCase & concatenated field pattern: WordNo -> Word No, CodeID -> Code ID
        line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)
        line = re.sub(r"([A-Za-z0-9]{2,})(No|Code|ID|Num|Val|Ref|Date|Type|Name|Amt)\b", r"\1 \2", line, flags=re.IGNORECASE)

        normalized = re.sub(r"[ \t]+", " ", line).strip()
        if normalized:
            lines.append(normalized)
    return "\n".join(lines).strip()


def score_ocr_text(text: str) -> float:
    text = normalize_ocr_text(text)
    if not text:
        return 0.0
    alnum_count = sum(1 for char in text if char.isalnum())
    word_count = len(re.findall(r"[A-Za-z0-9]{2,}", text))
    line_count = len([line for line in text.splitlines() if line.strip()])
    return alnum_count + (word_count * 4) + (line_count * 2)


def image_to_png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def load_image_from_bytes(image_bytes: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def build_ocr_image_variants(image: Image.Image) -> List[Image.Image]:
    """Create generic OCR-friendly variants for scans, screenshots, and photos."""
    image = image.convert("RGB")
    variants = [image]

    max_dimension = max(image.size)
    if max_dimension < 1600:
        scale = min(3.0, 1600 / max(1, max_dimension))
        resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        variants.append(resized)

    grayscale = variants[-1].convert("L")
    variants.append(grayscale)

    autocontrast = ImageOps.autocontrast(grayscale)
    variants.append(autocontrast)

    threshold = autocontrast.point(lambda pixel: 255 if pixel > 180 else 0)
    variants.append(threshold)

    unique_variants: List[Image.Image] = []
    seen = set()
    for variant in variants:
        signature = (variant.mode, variant.size)
        if signature in seen:
            continue
        seen.add(signature)
        unique_variants.append(variant)
    return unique_variants


def run_rapidocr(image_bytes: bytes) -> str:
    if RapidOCR is not None:
        try:
            engine = get_rapidocr_engine()
            result, _ = engine(image_bytes)
            if result:
                texts = [res[1] for res in result if res and len(res) > 1]
                return normalize_ocr_text("\n".join(texts))
        except Exception:
            pass
    return ""


def run_easyocr(image: Image.Image) -> str:
    if easyocr is not None:
        try:
            reader = get_easyocr_reader()
            results = reader.readtext(np.array(image.convert("RGB")), detail=0, paragraph=False)
            return normalize_ocr_text("\n".join(str(result) for result in results))
        except Exception:
            pass
    return ""


def run_tesseract(image: Image.Image) -> str:
    if pytesseract is not None:
        try:
            try:
                text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
            except Exception:
                text = pytesseract.image_to_string(image)
            return normalize_ocr_text(text)
        except Exception:
            pass
    return ""


def run_docling_on_image_bytes(image_bytes: bytes) -> str:
    if DocumentConverter is not None:
        try:
            converter = get_docling_converter()
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            try:
                res = converter.convert(tmp_path)
                doc_text = res.document.export_to_markdown()
                return normalize_ocr_text(doc_text)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception:
            pass
    return ""


def run_ocr_on_png_bytes(png_bytes: bytes) -> Tuple[str, Optional[str]]:
    """Run OCR using multiple engines and image preprocessing, returning the best generic result and engine name."""
    image = load_image_from_bytes(png_bytes)
    if image is None:
        return "", None

    candidates: List[Tuple[float, str, str]] = []
    for variant in build_ocr_image_variants(image):
        variant_bytes = image_to_png_bytes(variant)

        engine_map = {
            "rapidocr": lambda: run_rapidocr(variant_bytes),
            "easyocr": lambda: run_easyocr(variant),
            "tesseract": lambda: run_tesseract(variant),
            "docling": lambda: run_docling_on_image_bytes(variant_bytes),
        }

        for engine_name in OCR_ENGINE_ORDER:
            engine_name = engine_name.lower()
            if engine_name not in engine_map:
                continue
            try:
                text = engine_map[engine_name]()
            except Exception:
                logger.exception("OCR engine %s failed", engine_name)
                text = ""
            text = normalize_ocr_text(text)
            if len(text) >= MIN_OCR_TEXT_LENGTH:
                score = score_ocr_text(text) * ENGINE_BOOST.get(engine_name, 1.0)
                candidates.append((score, text, engine_name))

    if not candidates:
        return "", None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_text, best_engine = candidates[0]
    return best_text, best_engine


def run_ocr_on_pixmap(pixmap: fitz.Pixmap) -> Tuple[str, Optional[str]]:
    """Perform OCR on a PyMuPDF pixmap rendering and return (text, engine)."""
    try:
        png_bytes = pixmap.tobytes("png")
        return run_ocr_on_png_bytes(png_bytes)
    except Exception:
        return "", None


def run_ocr_on_image_file(image_path: Path) -> Tuple[str, Optional[str]]:
    """Perform OCR on a standalone image file (.png, .jpg, .jpeg), returning (text, engine)."""
    try:
        image_path = Path(image_path)
        img = Image.open(image_path)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return run_ocr_on_png_bytes(img_bytes.getvalue())
    except Exception:
        return "", None


def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, object]]:
    """
    Extract text from PDF pages using Hybrid Layout-Aware Targeted OCR:
    1. Extracts native digital text with bounding box coordinates.
    2. Identifies embedded images & scanned regions, performing targeted OCR on image boundaries.
    3. Reconstructs reading order layout by sorting native text + image OCR blocks spatially.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)
    pages: List[Dict[str, object]] = []

    for page_number in range(document.page_count):
        page = document.load_page(page_number)
        is_ocr = False
        page_ocr_engine = None

        text_blocks = page.get_text("blocks")
        image_list = page.get_images()
        has_images = len(image_list) > 0

        layout_elements: List[Dict[str, object]] = []

        native_text_parts = []
        for b in text_blocks:
            x0, y0, x1, y1, text, block_no, block_type = b[:7]
            clean_t = text.strip()
            if clean_t:
                native_text_parts.append(clean_t)
                layout_elements.append({
                    "y0": y0,
                    "x0": x0,
                    "type": "native_text",
                    "text": clean_t
                })

        native_text_full = "\n\n".join(native_text_parts).strip()

        if not native_text_full or len(native_text_full) < 20 or has_images:
            ocr_snippets: List[Tuple[float, float, str, Optional[str]]] = []

            for img_info in image_list:
                try:
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    img_y0 = rects[0].y0 if rects else 0.0
                    img_x0 = rects[0].x0 if rects else 0.0

                    base_image = document.extract_image(xref)
                    img_bytes = base_image.get("image")
                    if img_bytes:
                        img_ocr_text, img_ocr_engine = run_ocr_on_png_bytes(img_bytes)
                        if img_ocr_text and len(img_ocr_text.strip()) > 0:
                            ocr_snippets.append((img_y0, img_x0, img_ocr_text.strip(), img_ocr_engine))
                except Exception:
                    pass

            if not ocr_snippets and (not native_text_full or len(native_text_full) < 20):
                try:
                    pixmap = page.get_pixmap(dpi=OCR_DPI, alpha=False)
                    pm_text, pm_engine = run_ocr_on_pixmap(pixmap)
                    if pm_text and len(pm_text.strip()) > 0:
                        ocr_snippets.append((0.0, 0.0, pm_text.strip(), pm_engine))
                except Exception:
                    pass

            if ocr_snippets:
                is_ocr = True
                engine_counts: Dict[str, int] = {}
                for y0, x0, ocr_t, eng in ocr_snippets:
                    layout_elements.append({
                        "y0": y0,
                        "x0": x0,
                        "type": "image_ocr",
                        "text": ocr_t,
                        "engine": eng
                    })
                    eng_name = eng or "rapidocr"
                    engine_counts[eng_name] = engine_counts.get(eng_name, 0) + len(ocr_t)

                page_ocr_engine = max(engine_counts.items(), key=lambda kv: kv[1])[0] if engine_counts else "rapidocr"

        if not layout_elements:
            continue

        layout_elements.sort(key=lambda item: (round(float(item["y0"]), 1), round(float(item["x0"]), 1)))

        combined_page_lines = [str(el["text"]) for el in layout_elements if str(el["text"]).strip()]
        reconstructed_page_text = "\n\n".join(combined_page_lines).strip()

        cleaned_text = remove_noise_from_text(reconstructed_page_text)
        if not cleaned_text:
            cleaned_text = reconstructed_page_text

        entry = {
            "page_number": page_number + 1,
            "text": cleaned_text,
            "is_ocr": is_ocr,
            "document_name": pdf_path.name,
        }
        if is_ocr:
            entry["ocr_engine"] = page_ocr_engine
        pages.append(entry)

    document.close()
    return pages


def extract_docx_pages(docx_path: Path) -> List[Dict[str, object]]:
    """Extract text from Word .docx documents."""
    docx_path = Path(docx_path)
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX not found: {docx_path}")

    if docx is None:
        raise ImportError("python-docx is required to parse .docx files. Install it via pip install python-docx")

    doc = docx.Document(docx_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if not full_text.strip():
        return []

    return [
        {
            "page_number": 1,
            "text": full_text.strip(),
            "is_ocr": False,
            "document_name": docx_path.name,
        }
    ]


def extract_txt_pages(txt_path: Path) -> List[Dict[str, object]]:
    """Extract text from plain text (.txt, .md) files."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Text file not found: {txt_path}")

    content = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return []

    return [
        {
            "page_number": 1,
            "text": content,
            "is_ocr": False,
            "document_name": txt_path.name,
        }
    ]


def extract_image_file_pages(image_path: Path) -> List[Dict[str, object]]:
    """Extract text from standalone image files (.png, .jpg, .jpeg) using OCR."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    text, engine = run_ocr_on_image_file(image_path)
    if not text or not text.strip():
        return []

    return [
        {
            "page_number": 1,
            "text": text.strip(),
            "is_ocr": True,
            "ocr_engine": engine or "rapidocr",
            "document_name": image_path.name,
        }
    ]


def extract_document_pages(file_path: Path) -> List[Dict[str, object]]:
    """Unified document parser supporting PDF, DOCX, TXT, MD, PNG, JPG, JPEG."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_pages(file_path)
    elif suffix == ".docx":
        return extract_docx_pages(file_path)
    elif suffix in {".txt", ".md"}:
        return extract_txt_pages(file_path)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        return extract_image_file_pages(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
