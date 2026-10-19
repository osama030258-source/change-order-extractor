import pytest

from co_extract.schema import parse_amount, parse_date, parse_days


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$ 2,310.00", 2310.0),
        ("(12,136.90)", -12136.9),
        ("<2,500.00>", -2500.0),
        ("-$9,214.80", -9214.8),
        ("$-5.50", -5.5),
        ("1,808,617.98", 1808617.98),
        (126840.68, 126840.68),
        (None, None),
        ("", None),
    ],
)
def test_parse_amount(raw, expected):
    value, flags = parse_amount(raw)
    assert value == expected
    assert not any(f.startswith("error:") for f in flags)


def test_parse_amount_unparseable_and_ambiguous():
    assert parse_amount("TBD")[1] == ["error:amount_unparseable"]
    assert "warn:amount_multiple_numbers" in parse_amount("$1,000 to $2,000")[1]


def test_parse_date_formats():
    assert parse_date("May 12, 2005") == ("2005-05-12", [])
    assert parse_date("2009-01-22") == ("2009-01-22", [])
    assert parse_date("05/13/2025") == ("2025-05-13", [])  # 13 can only be a day
    assert parse_date("Effective January 22, 2009"[10:]) == ("2009-01-22", [])


def test_parse_date_flags_ambiguity_and_refuses_incomplete():
    iso, flags = parse_date("05/12/2025")
    assert iso == "2025-05-12" and "warn:ambiguous_date_format" in flags
    assert parse_date("March 2005") == (None, ["error:date_incomplete"])
    assert parse_date("garbage")[0] is None


def test_parse_days():
    assert parse_days(5) == (5, [])
    assert parse_days("Days: 0") == (0, [])
    assert parse_days("Add 60 working days") == (60, [])
    assert parse_days("n/a")[1] == ["error:days_unparseable"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CO #5R", "5R"),
        ("Change Order No. 3", "3"),
        ("Contract Modification No.: 4", "4"),
        ("CO-014", "014"),
        ("No.: 7", "7"),
        ("COR #11", "11"),
        ("5R", "5R"),
        ("09", "09"),
        ("CO", "CO"),
    ],
)
def test_clean_co_number(raw, expected):
    from co_extract.schema import clean_co_number

    assert clean_co_number(raw) == expected


def test_normalize_value_cleans_change_order_number():
    from co_extract.schema import normalize_value

    assert normalize_value("change_order_number", "CO #5R") == ("5R", [])
