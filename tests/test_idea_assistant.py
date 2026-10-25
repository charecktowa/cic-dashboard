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
                # La búsqueda tiene trabajos, pero la etiqueta promete algo que no: tampoco.
                {"tema": "Imágenes dentales", "busqueda": "segmentación de imágenes médicas"},
                {"tema": "De la", "busqueda": "segmentación de imágenes médicas"},
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


WORKS = ["Tesis · Segmentación de radiografías", "Artículo · Clasificación de imágenes (2024)"]


def test_advisor_summary_cites_the_given_works_by_number(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Trabaja imágenes médicas [2] y dirigió una tesis cercana [1]."
    asked = answering(monkeypatch, {"resumen": text, "obra": 2})
    summary = idea_assistant.advisor_summary("diagnóstico dental", "Persona 1", WORKS)
    assert summary == idea_assistant.Summary(text, 2)
    ((instructions, prompt),) = asked
    # Las obras van numeradas desde 1, en el orden en que se muestran.
    assert "[1] Tesis · Segmentación de radiografías" in prompt
    assert "[2] Artículo · Clasificación de imágenes (2024)" in prompt
    assert "3 a 5 oraciones" in instructions


@pytest.mark.parametrize(
    "answer",
    [
        # Cita una obra que no está en la lista.
        {"resumen": "Trabaja imágenes [3].", "obra": 1},
        # No cita ninguna.
        {"resumen": "Trabaja imágenes médicas.", "obra": 1},
        # Una cita válida no salva a otra con otro formato.
        {"resumen": "Trabaja imágenes [1] y radiografías [2, 99].", "obra": 1},
        {"resumen": "Trabaja imágenes [1] y radiografías [1-3].", "obra": 1},
        {"resumen": "Trabaja imágenes [1] y radiografías [obra 2].", "obra": 1},
        # Otras cifras Unicode tampoco: la lista se numera con 1, 2, 3.
        {"resumen": "Trabaja imágenes [1] y radiografías [٢].", "obra": 1},
        # La obra elegida no existe, o no es un número.
        {"resumen": "Trabaja imágenes [1].", "obra": 5},
        {"resumen": "Trabaja imágenes [1].", "obra": "1"},
        {"resumen": "Trabaja imágenes [1].", "obra": True},
        {"resumen": "", "obra": 1},
        {"resumen": "x [1] " * 200, "obra": 1},
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
