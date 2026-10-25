"""Ayudas de Qwen3.7 Flash para Encuentra tu lab, siempre verificadas contra el catálogo.

El modelo propone; el catálogo decide. Una sugerencia de área vecina solo se muestra si sus
palabras aparecen en títulos del CIC, y un resumen solo puede señalar una obra que se le dio.
"""

import re
from dataclasses import dataclass

import text
from dashscope_client import chat_json
from semantic_search import Corpus

# Una sugerencia sirve si al menos un par de palabras suyo ("imágenes médicas") aparece en
# MIN_TITLES títulos o más. Palabras sueltas no bastan: "análisis" o "desarrollo" están en
# cientos de títulos y dejarían pasar cualquier cosa.
MIN_TITLES = 3
MAX_SUGGESTIONS = 3
MAX_TOPIC_CHARS = 60
MAX_QUERY_CHARS = 200

NEIGHBOR_INSTRUCTIONS = """\
Ayudas a estudiantes a buscar asesor en el catálogo de títulos de un centro de investigación \
en computación (inteligencia artificial, visión por computadora, lenguaje natural, robótica, \
redes, ciberseguridad, cómputo cuántico, señales, sistemas embebidos).
El catálogo no tiene trabajos con algunas palabras de la idea del estudiante. Propón hasta 3 \
áreas de computación cercanas a la idea donde sí es probable encontrar trabajos, conservando \
lo esencial de la idea. Las búsquedas no deben usar las palabras sin trabajos.
Responde solo un objeto JSON con esta forma:
{"sugerencias": [{"tema": "2 a 5 palabras", "busqueda": "la idea reformulada en una oración, \
en español"}]}
No inventes trabajos, personas ni laboratorios."""


@dataclass(frozen=True)
class Suggestion:
    """Un área vecina: un nombre corto para el botón y la búsqueda que lanza."""

    topic: str
    query: str


def covered_pairs(query: str, corpus: Corpus) -> list[str]:
    """Los pares de palabras de la búsqueda que aparecen en MIN_TITLES títulos o más."""
    frequency = corpus.document_frequency
    return [
        term for term in text.terms(query) if " " in term and frequency.get(term, 0) >= MIN_TITLES
    ]


def neighbor_areas(idea: str, missing: list[str], corpus: Corpus) -> list[Suggestion]:
    """Hasta MAX_SUGGESTIONS áreas cercanas a la idea que sí tienen trabajos en el catálogo.

    Lanza dashscope_client.ChatError si el modelo no responde; una respuesta con otra forma,
    o sugerencias sin cobertura, simplemente no dan sugerencias.
    """
    answer = chat_json(
        NEIGHBOR_INSTRUCTIONS,
        f"Idea: {idea}\nPalabras sin trabajos en el catálogo: {', '.join(missing)}",
    )
    uncovered = {term for word in missing for term in text.terms(word)}
    proposals = answer.get("sugerencias")
    if not isinstance(proposals, list):
        return []
    suggestions: list[Suggestion] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        topic, query = proposal.get("tema"), proposal.get("busqueda")
        if not isinstance(topic, str) or not isinstance(query, str):
            continue
        topic, query = topic.strip(), query.strip()
        if not topic or not query or len(topic) > MAX_TOPIC_CHARS or len(query) > MAX_QUERY_CHARS:
            continue
        # Sin las palabras que no tienen trabajos, y con al menos un par que sí los tiene.
        if not set(text.terms(query)) & uncovered and covered_pairs(query, corpus):
            suggestions.append(Suggestion(topic, query))
        if len(suggestions) == MAX_SUGGESTIONS:
            break
    return suggestions


SUMMARY_INSTRUCTIONS = """\
Explicas a un estudiante por qué una persona asesora del CIC podría encajar con su idea de \
tesis, usando solo la lista numerada de sus obras que se te da.
Responde solo un objeto JSON con esta forma:
{"resumen": "3 a 5 oraciones en español, dirigidas al estudiante; cita las obras que apoyan \
cada afirmación con su número entre corchetes, como [1] o [2][3]", \
"obra": el número de la obra que más se relaciona con la idea}
No copies los títulos completos: cítalos por su número. No inventes obras, cargos, fechas ni \
datos que no estén en los títulos. Si la relación es parcial, dilo."""
MAX_SUMMARY_CHARS = 900
CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Summary:
    """Por qué una persona encaja con la idea, citando sus obras por número (1 = la primera).

    `work` es el número de la obra que mejor lo muestra.
    """

    text: str
    work: int


def advisor_summary(idea: str, person: str, works: list[str]) -> Summary | None:
    """Un resumen de por qué `person` encaja con la idea, citando `works` por número.

    `works` son títulos tal como se le muestran al visitante ("Tesis · título (año)"), y se
    numeran desde 1. El resumen debe citar al menos una obra, y toda cita y la obra elegida
    deben ser números de la lista; si no, o si la respuesta tiene otra forma, None. Lanza
    dashscope_client.ChatError si el modelo no responde.
    """
    if not works:
        return None
    listed = "\n".join(f"[{number}] {work}" for number, work in enumerate(works, start=1))
    answer = chat_json(SUMMARY_INSTRUCTIONS, f"Idea: {idea}\nPersona: {person}\nObras:\n{listed}")
    summary, chosen = answer.get("resumen"), answer.get("obra")
    # bool es un int en Python: true no es una obra.
    if not isinstance(summary, str) or type(chosen) is not int:
        return None
    summary = summary.strip()
    cited = {int(number) for number in CITATION.findall(summary)}
    valid = set(range(1, len(works) + 1))
    if not summary or len(summary) > MAX_SUMMARY_CHARS or not cited or not cited <= valid:
        return None
    return Summary(summary, chosen) if chosen in valid else None
