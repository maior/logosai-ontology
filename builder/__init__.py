"""
Ontology Builder Framework — arbitrary data → ontology.

데이터(파일/폴더) + 온톨로지 프레임워크 + LLM을 연결해, 임의의 데이터를
분석하고 네임스페이스별 온톨로지로 구성한다.

Usage:
    from ontology.builder import OntologyBuilder, BuilderSchema

    builder = OntologyBuilder(
        schema=BuilderSchema.preset_document(),  # or from_dict(custom)
        namespace="insurance",
    )
    report = await builder.build_from_folder("docs/약관/")
"""

from .models import BuilderSchema, BuildReport, Chunk, Extraction
from .pipeline import OntologyBuilder
from .readers import SUPPORTED_EXTENSIONS, UnsupportedFormatError, read_file, read_folder
from .segmenter import segment
from .validator import clean_extraction

__all__ = [
    "OntologyBuilder",
    "BuilderSchema",
    "BuildReport",
    "Chunk",
    "Extraction",
    "read_file",
    "read_folder",
    "segment",
    "clean_extraction",
    "SUPPORTED_EXTENSIONS",
    "UnsupportedFormatError",
]
