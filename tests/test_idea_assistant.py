"""Las ayudas del chat se verifican contra el catálogo antes de mostrarse."""

from typing import Any

import pytest

import idea_assistant
from semantic_search import Corpus

CORPUS = Corpus.of(
    ["Segmentación de imágenes médicas con redes neuronales"] * 3
    + ["Clasificación de radiografías con aprendizaje profundo"] * 3
)


def answering(monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any]) -> list[tuple[str, str]]:
    """El chat responde `answer`; devuelve lo que se le preguntó."""
    asked: list[tuple[str, str]] = []

    def chat_json(instructions: str, prompt: str) -> dict[str, Any]:
        asked.append((instructions, prompt))
        return answer

    monkeypatch.setattr(idea_assistant, "chat_json", chat_json)
    return asked


def test_neighbor_areas_keep_only_what_the_catalog_covers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = answering(
        monkeypatch,
        {
            "sugerencias": [
                {"tema": "Imágenes médicas", "busqueda": "segmentación de imágenes médicas"},
                # El catálogo no tiene trabajos de esto: no se sugiere.
                {"tema": "Prótesis dentales", "busqueda": "diseño de prótesis dentales"},
                {"tema": "Radiografías", "busqueda": "clasificación de radiografías"},
                # Repite la palabra sin trabajos: tampoco.
                {
                    "tema": "Odontología",
                    "busqueda": "segmentación de imágenes médicas en odontología",
                },
                # Solo palabras sueltas comunes, ningún par con títulos: tampoco.
                {"tema": "Análisis", "busqueda": "clasificación con redes y segmentación"},
            ]
        },
    )
    suggestions = idea_assistant.neighbor_areas(
        "deep learning para odontología", ["odontologia"], CORPUS
    )
    assert [s.topic for s in suggestions] == ["Imágenes médicas", "Radiografías"]
    assert suggestions[0].query == "segmentación de imágenes médicas"
    ((_, prompt),) = asked
    assert "deep learning para odontología" in prompt and "odontologia" in prompt


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"sugerencias": "imágenes médicas"},
        {"sugerencias": ["imágenes médicas"]},
        {"sugerencias": [{"tema": 3, "busqueda": "segmentación de imágenes médicas"}]},
        {"sugerencias": [{"tema": "x" * 61, "busqueda": "segmentación de imágenes médicas"}]},
        {"sugerencias": [{"tema": "Imágenes", "busqueda": "imágenes " * 30}]},
    ],
)
def test_neighbor_areas_ignore_answers_of_another_shape(
    monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any]
) -> None:
    answering(monkeypatch, answer)
    assert idea_assistant.neighbor_areas("idea", ["odontologia"], CORPUS) == []


def test_neighbor_areas_offer_at_most_three(monkeypatch: pytest.MonkeyPatch) -> None:
    proposal = {"tema": "Imágenes médicas", "busqueda": "segmentación de imágenes médicas"}
    answering(monkeypatch, {"sugerencias": [proposal] * 6})
    assert len(idea_assistant.neighbor_areas("idea", ["odontologia"], CORPUS)) == 3


WORKS = ["Tesis · Segmentación de radiografías", "Artículo · Clasificación de imágenes médicas"]


@pytest.mark.parametrize("chosen", [WORKS[1], "Clasificación de imágenes médicas"])
def test_advisor_summary_points_to_one_of_the_given_works(
    monkeypatch: pytest.MonkeyPatch, chosen: str
) -> None:
    asked = answering(monkeypatch, {"resumen": "Trabaja imágenes médicas.", "obra": chosen})
    summary = idea_assistant.advisor_summary("diagnóstico dental", "Persona 1", WORKS)
    assert summary == idea_assistant.Summary("Trabaja imágenes médicas.", WORKS[1])
    ((_, prompt),) = asked
    assert "diagnóstico dental" in prompt and all(work in prompt for work in WORKS)


@pytest.mark.parametrize(
    "answer",
    [
        # Una obra inventada: el resumen no se muestra.
        {"resumen": "Trabaja imágenes.", "obra": "Detección de caries con IA"},
        {"resumen": "", "obra": WORKS[0]},
        {"resumen": "x" * 601, "obra": WORKS[0]},
        {"resumen": ["Trabaja"], "obra": WORKS[0]},
        {},
    ],
)
def test_advisor_summary_drops_answers_it_cannot_trust(
    monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any]
) -> None:
    answering(monkeypatch, answer)
    assert idea_assistant.advisor_summary("idea", "Persona 1", WORKS) is None


def test_advisor_summary_needs_works(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = answering(monkeypatch, {})
    assert idea_assistant.advisor_summary("idea", "Persona 1", []) is None
    assert asked == []
