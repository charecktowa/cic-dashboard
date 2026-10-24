"""Cómo se parte un texto en términos, igual en export.py y en la app.

Un término es una palabra de contenido (sin acentos ni palabras vacías) o un
par de palabras de contenido consecutivas, como "redes neuronales".
"""

import re
import unicodedata

import polars as pl

STOPWORD_TEXT = """
    a an and are as at be by for from in into is it its of on or the their this to with
    using use used based via toward towards new over under between through within
    de la el en y los las del para por con un una unos unas que se al su sus como
    mediante basado basada basados basadas sobre entre e o u lo uso usando desde hacia
    sin le este esta estos estas es son nuevo nueva nuevos nuevas
    me mi mis quiero quisiera gustaria trabajar tesis tema temas interesa interesan algo
"""
STOPWORDS = frozenset(STOPWORD_TEXT.split())
WORD = r"[a-z0-9]+(?:-[a-z0-9]+)*"


def words_expr(column: str) -> pl.Expr:
    """Palabras en minúsculas y sin acentos de una columna de texto."""
    return (
        pl.col(column)
        .str.to_lowercase()
        .str.normalize("NFKD")
        .str.replace_all(r"\p{M}", "")
        .str.extract_all(WORD)
    )


def is_content(word: str) -> bool:
    return word not in STOPWORDS and len(word) >= 3 and not word.isdigit()


def words(text: str) -> list[str]:
    """Palabras en minúsculas y sin acentos de un texto."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return re.findall(WORD, plain)


def terms(text: str) -> list[str]:
    """Palabras de contenido y pares consecutivos de ellas, sin repetir."""
    content = [w for w in words(text) if is_content(w)]
    pairs = [f"{a} {b}" for a, b in zip(content, content[1:], strict=False)]
    return list(dict.fromkeys(content + pairs))
