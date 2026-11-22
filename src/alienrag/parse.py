"""PDF -> DoclingDocument JSON (+ exported picture PNGs)."""

from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode

from alienrag import config


def build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions(
        do_ocr=False,  # Dell manuals are born-digital (verified: every page has a text layer)
        do_table_structure=True,
        generate_picture_images=True,
        images_scale=config.IMAGES_SCALE,
    )
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def parsed_path(pdf: Path) -> Path:
    return config.PARSED_DIR / f"{pdf.stem}.json"


def parse_pdf(converter: DocumentConverter, pdf: Path) -> DoclingDocument:
    doc = converter.convert(pdf).document
    out = parsed_path(pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    img_dir = config.IMAGES_DIR / pdf.stem
    img_dir.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(out, artifacts_dir=img_dir, image_mode=ImageRefMode.REFERENCED)
    return doc


def load_parsed(pdf: Path) -> DoclingDocument:
    return DoclingDocument.load_from_json(parsed_path(pdf))
