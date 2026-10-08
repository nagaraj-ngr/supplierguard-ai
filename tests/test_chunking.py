import re

import pytest

from supplierguard.ingest.chunking import chunk_text

DOC = "\n\n".join(
    f"Section {i}\n" + " ".join(f"This is sentence {j} of section {i} with some filler words." for j in range(6))
    for i in range(8)
)


def test_chunks_respect_max_size_and_are_never_empty():
    chunks = chunk_text(DOC, max_chars=400, overlap=60)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 400 for c in chunks)


def test_no_content_is_lost():
    chunks = chunk_text(DOC, max_chars=400, overlap=60)
    joined = " ".join(chunks)
    assert set(re.findall(r"\w+", DOC)) <= set(re.findall(r"\w+", joined))


def test_overlap_is_a_word_aligned_suffix_of_the_previous_chunk():
    chunks = chunk_text(DOC, max_chars=800, overlap=100)
    overlapped = 0
    for a, b in zip(chunks, chunks[1:]):
        tail = b.split("\n\n")[0]
        if a.endswith(tail) and len(tail) < len(a):
            overlapped += 1
            assert len(tail) <= 100
            assert a[len(a) - len(tail) - 1].isspace(), "overlap started in the middle of a word"
    assert overlapped >= 1


def test_overlap_is_dropped_rather_than_exceeding_the_size_limit():
    # Units of ~370 chars with max 400 leave no room for an overlap tail.
    chunks = chunk_text(DOC, max_chars=400, overlap=100)
    assert all(len(c) <= 400 for c in chunks)


def test_short_text_is_a_single_chunk():
    assert chunk_text("Hello world.", max_chars=800) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == [] and chunk_text("  \n\n  ") == []


def test_oversized_paragraph_is_split_on_sentences():
    para = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(para, max_chars=200, overlap=0)
    assert all(len(c) <= 200 for c in chunks) and len(chunks) > 5
    assert all(c.endswith(".") for c in chunks)


def test_single_giant_token_is_hard_split():
    chunks = chunk_text("x" * 1000, max_chars=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks) and "".join(chunks) == "x" * 1000


def test_deterministic():
    assert chunk_text(DOC, 300, 50) == chunk_text(DOC, 300, 50)


@pytest.mark.parametrize("max_chars,overlap", [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)])
def test_invalid_parameters_rejected(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=max_chars, overlap=overlap)
