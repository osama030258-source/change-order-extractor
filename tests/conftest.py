import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
LABELS = ROOT / "data" / "labels"


def fld(value, evidence, conf=0.9):
    return {"value": value, "evidence": evidence, "self_confidence": conf}


NULL = {"value": None, "evidence": None, "self_confidence": None}


def glenwood_response(hallucinate: bool = False) -> str:
    """Canned model output for the Glenwood fixture. Test input only; it says nothing about real accuracy."""
    co5r = {
        "change_order_number": fld("5R", "Change Order CO #5R Distribution List:"),
        "date": fld("05/13/2025", "Date : 05/13/2025"),
        "project_name": fld("Glenwood Facades Phase 1", "Glenwood Facades Phase 1 Owner___x____"),
        "contractor": fld("Cornerstone Commercial Contractors", "Contractor : Cornerstone Commercial Contractors"),
        "description": fld("Remove and replace deteriorated brick at 424", "Remove and replace deteriorated brick with close"),
        "cost_change": fld("$ 2,310.00", "TOTAL  $ 2,310.00"),
        "schedule_impact_days": fld(0, "(unchanged) by Days: 0"),
        "requested_by": fld("Cornerstone Commercial Contractors", "Cornerstone Commercial Contractors, Inc. Would like to request a change order"),
        "approved_by": NULL,
        "status": NULL,
        "original_contract_sum": fld("$ 815,550.00", "was  $ 815,550.00"),
        "contract_sum_before_change": NULL,
        "new_contract_sum": fld("$ 826,669.50", "will be $ 826,669.50"),
        "line_items": [
            {"description": "Masonry remove and replace 50 brick", "amount": "$ 2,000.00", "evidence": "Masonry remove and replace 50 brick $ 2,000.00"},
            {"description": "Dumpster", "amount": "$ 100.00", "evidence": "Dumpster  $ 100.00"},
            {"description": "General Conditions", "amount": "$ 210.00", "evidence": "General Conditions  $ 210.00"},
        ],
    }
    co9 = {
        "change_order_number": fld("09", "CO #09 Distribution List:"),
        "date": fld("05/12/2025", "Date : 05/12/2025"),
        "project_name": fld("Glenwood Facades Phase 1", "Glenwood Facades Phase 1 Owner___x____"),
        "contractor": fld("Cornerstone Commercial Contractors", "Contractor : Cornerstone Commercial Contractors"),
        "description": fld("Remove electrical from 409 Sharp St.", "cost to remove electrical from 409 project"),
        "cost_change": fld("$ 0.00", "in the amount of  $ 0.00"),
        "schedule_impact_days": fld(0, "(unchanged) by Days: 0"),
        "requested_by": fld("Cornerstone Commercial Contractors", "Cornerstone Commercial Contractors, Inc. would like to request a change order"),
        "approved_by": fld("J. Miller", "Approved by J. Miller on 05/14/2025", 0.95) if hallucinate else NULL,
        "status": NULL,
        "original_contract_sum": fld("$ 815,550.00", "was  $ 815,550.00"),
        "contract_sum_before_change": NULL,
        "new_contract_sum": fld("$ 826,669.50", "will be $ 826,669.50"),
        "line_items": [{"description": "Electrical", "amount": "<2,500.00>", "evidence": "Electrical  <2,500.00>"}],
    }
    return json.dumps({"change_orders": [co5r, co9]})


@pytest.fixture
def glenwood_path():
    return SAMPLES / "glenwood_2025_cos.txt"
