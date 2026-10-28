"""Vincula al personal investigador del CIC con el padrón del SNII de SECIHTI.

El padrón se publica en datos.gob.mx (CC BY 4.0), pero su servidor rechaza las descargas
que no vienen de un navegador. Por eso se descarga a mano y este script solo lo lee:

    https://www.datos.gob.mx/dataset/sistema_nacional_investigadoras_investigadores_snii_s191
    uv run python snii.py ~/Downloads/s191_snii_2s_2025.csv

Corre en tu máquina, después de export.py; el VPS nunca lee el padrón.
"""

import os
import re
import sys
from pathlib import Path

import polars as pl

from text import words

DATA = Path(os.environ.get("CIC_DATA_DIR", Path(__file__).parent / "data"))

IPN = "Instituto Politecnico Nacional"
LEVELS = {"C": "Candidato", "1": "Nivel I", "2": "Nivel II", "3": "Nivel III", "E": "Emérito"}
# El padrón marca sin fin a quien no vence, como las personas eméritas: 2099 o más.
NO_END = 2099
# "s191_snii_2s_2025.csv" es el corte del segundo semestre de 2025.
CUT = re.compile(r"(\d)s_(\d{4})")


def cut_label(path: Path) -> str:
    """El corte del padrón según el nombre del archivo, como "2.º semestre de 2025"."""
    found = CUT.search(path.name)
    if not found:
        raise SystemExit(f"No reconozco el corte en {path.name}; se espera algo como 2s_2025")
    half, year = found.groups()
    return f"{half}.º semestre de {year}"


def match(people: list[str], roster: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """El registro del SNII de cada persona, si es uno solo; y quiénes tienen varios.

    Busca dentro del IPN, porque el padrón no distingue al CIC de los demás centros. Una
    persona coincide con un registro si todas las palabras de su nombre están en él, sin
    acentos: así "Grigori Sidorov" encuentra "Grigori Sidorov Sin Dato". Quien no aparece
    no queda en la tabla: el tablero nunca dice que alguien no está en el SNII.
    """
    ipn = roster.filter(pl.col("institucion_acreditacion_comision") == IPN)
    records = [
        (set(words(f"{r['nombre']} {r['apellido1']} {r['apellido2'] or ''}")), r)
        for r in ipn.iter_rows(named=True)
    ]
    found, ambiguous = [], []
    for person in people:
        name = set(words(person))
        hits = [r for record, r in records if name <= record]
        if len(hits) > 1:
            ambiguous.append(person)
        elif hits:
            r = hits[0]
            end = int(r["fin_vigencia"][:4]) if r["fin_vigencia"][:4].isdigit() else None
            found.append(
                {
                    "person": person,
                    "level": LEVELS[r["nivel"]],
                    "start": int(r["inicio_vigencia"][:4]),
                    "end": end if end and end < NO_END else None,
                }
            )
    schema = {"person": pl.String, "level": pl.String, "start": pl.Int32, "end": pl.Int32}
    return pl.DataFrame(found, schema=schema), ambiguous


def main(path: Path) -> None:
    people = pl.read_parquet(DATA / "people.parquet").drop_nulls("lab")["person"].to_list()
    roster = pl.read_csv(path, infer_schema_length=0)
    found, ambiguous = match(people, roster)
    found.with_columns(cut=pl.lit(cut_label(path))).write_parquet(DATA / "snii.parquet")
    print(f"{found.height} de {len(people)} personas en el SNII ({cut_label(path)}).")
    # Varios registros posibles: no se asigna ninguno; revisa a mano si hace falta.
    for person in ambiguous:
        print(f"  Varios registros, sin asignar: {person}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).expanduser())
