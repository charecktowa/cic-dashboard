import text


def test_words_drop_case_and_accents() -> None:
    assert text.words("Detección de Señales") == ["deteccion", "de", "senales"]


def test_terms_keep_content_words_and_their_pairs() -> None:
    terms = text.terms("Detección de emociones con deep learning")
    assert terms == [
        "deteccion",
        "emociones",
        "deep",
        "learning",
        "deteccion emociones",
        "emociones deep",
        "deep learning",
    ]


def test_terms_skip_stopwords_short_words_and_numbers() -> None:
    assert text.terms("Quiero una tesis de IA en 2026") == []


def test_terms_do_not_repeat() -> None:
    assert text.terms("robots robots") == ["robots", "robots robots"]
