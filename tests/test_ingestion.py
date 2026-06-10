from __future__ import annotations

from pathlib import Path

import fitz

from lit_review_assistant.pipeline.chunking import chunk_page_text, chunk_pages_with_sections
from lit_review_assistant.pipeline.pdf import (
    PageText,
    PaperMetadata,
    extract_pages,
    infer_paper_metadata_from_first_page,
    infer_paper_metadata_from_name,
)
from lit_review_assistant.pipeline.sections import detect_sections
from lit_review_assistant.services import ingest_pdf


def write_sample_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Abstract\nThis paper studies retrieval augmented literature review.\n\n"
        "Introduction\nPrior systems summarize papers but often lose provenance.\n",
    )
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Methods\nWe extract claims from chunks and preserve offsets.\n\n"
        "Results\nSentence-level traceability improves review inspection.\n",
    )
    doc.save(path)
    doc.close()


def test_extract_pages_from_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    write_sample_pdf(pdf_path)

    pages = extract_pages(pdf_path)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Abstract" in pages[0].text
    assert "Methods" in pages[1].text


def test_detect_sections_from_common_headings() -> None:
    pages = [
        PageText(1, "Abstract\nShort summary.\nIntroduction\nProblem framing.\n"),
        PageText(2, "Methods\nPipeline design.\nResults\nEvaluation.\n"),
    ]

    sections = detect_sections(pages)

    assert [section.normalized_type for section in sections] == [
        "abstract",
        "introduction",
        "methods",
        "results",
    ]
    assert sections[0].page_start == 1
    assert sections[-1].page_end == 2


def test_chunk_offsets_round_trip_to_source_text() -> None:
    page = PageText(1, "Introduction\n" + ("A sentence about methods. " * 20))

    chunks = chunk_page_text(page, max_chars=120, overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text == page.text[chunk.start_char : chunk.end_char]
        assert chunk.end_char > chunk.start_char


def test_chunks_receive_best_effort_section_metadata() -> None:
    pages = [PageText(1, "Introduction\n" + ("Background text. " * 10) + "\nMethods\n" + ("Method text. " * 10))]
    sections = detect_sections(pages)

    chunks = chunk_pages_with_sections(pages, sections, max_chars=90, overlap=10)

    assert {chunk.section_type for chunk in chunks if chunk.section_type} >= {"introduction", "methods"}


def test_infer_paper_metadata_from_filename() -> None:
    metadata = infer_paper_metadata_from_name(
        "Anyfantakis1 Baigl5-2015-Modulation of the Coffee-Ring Effect in Particle-Surfactant Mixtures.pdf"
    )

    assert metadata.authors == ["Anyfantakis", "Baigl"]
    assert metadata.year == 2015
    assert metadata.title == "Modulation of the Coffee-Ring Effect in Particle-Surfactant Mixtures"


def test_infer_paper_metadata_from_first_page_title_and_authors() -> None:
    metadata = infer_paper_metadata_from_first_page(
        "Modulation of the Coffee-Ring Effect in Particle/Surfactant Mixtures:\n"
        "the Importance of Particle-Interface Interactions\n"
        "Manos Anyfantakis,*,† Zheng Geng,† Mathieu Morel, Sergii Rudiuk, and Damien Baigl*\n"
        "Department of Chemistry, Ecole Normale Superieure-PSL Research University, Paris, France\n"
        "ABSTRACT: We study the effect of surfactants on deposits.\n"
    )

    assert metadata.title == (
        "Modulation of the Coffee-Ring Effect in Particle/Surfactant Mixtures: "
        "the Importance of Particle-Interface Interactions"
    )
    assert metadata.authors == [
        "Manos Anyfantakis",
        "Zheng Geng",
        "Mathieu Morel",
        "Sergii Rudiuk",
        "Damien Baigl",
    ]


def test_ingest_pdf_passes_chunk_settings(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF test placeholder")
    observed: dict[str, int] = {}

    class FakeSession:
        def scalar(self, _query):
            return None

        def add(self, _model):
            return None

        def flush(self):
            return None

    def fake_extract_pages(_path):
        return [PageText(1, "Introduction\nShort text.")]

    def fake_extract_pdf_metadata(_path):
        return PaperMetadata()

    def fake_detect_sections(_pages):
        return []

    def fake_chunk_pages_with_sections(_pages, _sections, max_chars, overlap):
        observed["max_chars"] = max_chars
        observed["overlap"] = overlap
        return []

    monkeypatch.setattr("lit_review_assistant.services.extract_pages", fake_extract_pages)
    monkeypatch.setattr("lit_review_assistant.services.extract_pdf_metadata", fake_extract_pdf_metadata)
    monkeypatch.setattr("lit_review_assistant.services.detect_sections", fake_detect_sections)
    monkeypatch.setattr("lit_review_assistant.services.chunk_pages_with_sections", fake_chunk_pages_with_sections)

    ingest_pdf(FakeSession(), pdf_path, chunk_max_chars=3_000, chunk_overlap=250)  # type: ignore[arg-type]

    assert observed == {"max_chars": 3_000, "overlap": 250}
