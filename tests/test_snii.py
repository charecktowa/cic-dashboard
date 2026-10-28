"""snii.py vincula por nombre dentro del IPN y nunca adivina."""

from pathlib import Path

import polars as pl
import pytest

import snii

IPN = snii.IPN


def roster(*rows: tuple[str, str, str, str, str, str, str]) -> pl.DataFrame:
    """Un padrón con las columnas que usa snii.py, todas como texto, como en el CSV."""
    names = ["nombre", "apellido1", "apellido2", "nivel", "inicio_vigencia", "fin_vigencia"]
    columns = [*names, "institucion_acreditacion_comision"]
    return pl.DataFrame([dict(zip(columns, row, strict=True)) for row in rows])


def test_match_keeps_one_record_per_person_without_accents() -> None:
    found, ambiguous = snii.match(
        ["Héctor Báez Medina", "Grigori Sidorov", "Persona Ausente"],
        roster(
            ("Hector", "Baez", "Medina", "1", "2024-01-01T00:00:00", "2027-12-31T00:00:00", IPN),
            # Un solo apellido: el padrón llena el segundo con "Sin Dato".
            (
                "Grigori",
                "Sidorov",
                "Sin Dato",
                "3",
                "2025-01-01T00:00:00",
                "2029-12-31T00:00:00",
                IPN,
            ),
        ),
    )
    assert found.rows() == [
        ("Héctor Báez Medina", "Nivel I", 2024, 2027),
        ("Grigori Sidorov", "Nivel III", 2025, 2029),
    ]
    # Quien no aparece no queda en la tabla: el tablero no dice que no está en el SNII.
    assert ambiguous == []


def test_match_looks_only_inside_the_ipn() -> None:
    found, _ = snii.match(
        ["Héctor Báez Medina"],
        roster(
            ("Hector", "Baez", "Medina", "2", "2024-01-01T00:00:00", "2027-12-31T00:00:00", "UNAM")
        ),
    )
    assert found.is_empty()


def test_match_leaves_out_people_with_several_records() -> None:
    found, ambiguous = snii.match(
        ["Ana Pérez"],
        roster(
            ("Ana", "Perez", "Lopez", "1", "2024-01-01T00:00:00", "2027-12-31T00:00:00", IPN),
            ("Ana", "Perez", "Ruiz", "2", "2023-01-01T00:00:00", "2026-12-31T00:00:00", IPN),
        ),
    )
    assert found.is_empty()
    assert ambiguous == ["Ana Pérez"]


@pytest.mark.parametrize("end", ["2099-12-31T00:00:00", "2121-12-31T00:00:00", "SIN DATO"])
def test_match_has_no_end_for_records_that_do_not_expire(end: str) -> None:
    found, _ = snii.match(
        ["Héctor Báez Medina"],
        roster(("Hector", "Baez", "Medina", "E", "2016-01-01T00:00:00", end, IPN)),
    )
    assert found.rows() == [("Héctor Báez Medina", "Emérito", 2016, None)]


def test_cut_label_comes_from_the_file_name() -> None:
    assert snii.cut_label(Path("s191_snii_2s_2025.csv")) == "2.º semestre de 2025"
    with pytest.raises(SystemExit):
        snii.cut_label(Path("padron.csv"))
