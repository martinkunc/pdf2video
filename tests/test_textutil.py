from pdf2video.textutil import (
    chunk_paragraphs,
    detect_language,
    lines_to_paragraphs,
    plan_parts,
    safe_filename,
    split_sentences,
)


def test_split_sentences_keeps_abbreviations():
    text = "Dr. Smith arrived at 5 p.m. today. He said hello! Did it work? Yes."
    assert split_sentences(text) == [
        "Dr. Smith arrived at 5 p.m. today.",
        "He said hello!",
        "Did it work?",
        "Yes.",
    ]


def test_lines_to_paragraphs_joins_wrapped_lines_and_hyphens():
    lines = ["This is a long line that wraps and con-", "tinues here.", "", "Second para.", "12"]
    assert lines_to_paragraphs(lines) == [
        "This is a long line that wraps and continues here.",
        "Second para.",
    ]


def test_chunks_respect_limit_and_sentence_boundaries(lorem):
    paragraphs = [lorem * 20, "Short one.", lorem * 3]
    chunks = chunk_paragraphs(paragraphs, max_chars=500)
    assert all(len(c.text) <= 500 for c in chunks)
    assert all(c.text.endswith((".", "!", "?")) for c in chunks)
    assert chunks[-1].ends_paragraph
    joined = " ".join(c.text for c in chunks).split()
    assert joined == " ".join(paragraphs).split()


def test_plan_parts_single_part_when_short():
    assert plan_parts([10, 20, 30], [True] * 3, 600) == [[0, 1, 2]]


def test_plan_parts_balanced_and_within_limit():
    durations = [60.0] * 25  # 25 minutes
    parts = plan_parts(durations, [True] * 25, 600)
    assert len(parts) == 3
    assert [i for p in parts for i in p] == list(range(25))
    assert all(sum(durations[i] for i in p) <= 600 for p in parts)
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 2  # 9/8/8, not 10/10/5


def test_plan_parts_never_exceeds_limit_with_uneven_chunks():
    durations = [37.0, 95.0, 12.0, 80.0, 64.0, 120.0, 5.0, 99.0, 88.0, 40.0] * 4
    ends = [i % 3 == 0 for i in range(len(durations))]
    parts = plan_parts(durations, ends, 300)
    assert [i for p in parts for i in p] == list(range(len(durations)))
    assert all(sum(durations[i] for i in p) <= 300 for p in parts)


def test_safe_filename():
    assert safe_filename('Part 1: "Why?" / How') == "Part 1 Why How"
    assert safe_filename("   ") == "Untitled"
    assert len(safe_filename("word " * 50)) <= 80


def test_detect_language():
    assert detect_language("The cat and the dog were in the garden with the kids.") == "en"
    assert detect_language("Pes a kočka jsou na zahradě, ale to není pro ně.") == "cs"
    assert detect_language("Der Hund und die Katze sind nicht mit dem Kind.") == "de"
