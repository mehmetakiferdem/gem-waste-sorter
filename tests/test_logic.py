import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from climate import ClimateLedger
from decision import EMPTY, UNSURE, WAIT, DecisionMaker
from prepare_dataset import bin_of

BINS = {"plastik": {"mass_kg": 0.03, "co2_kg_per_kg": 1.0},
        "cam": {"mass_kg": 0.3, "co2_kg_per_kg": 0.3},
        "diger": {"mass_kg": 0.05, "co2_kg_per_kg": 0.0}}


def feed(dm, n, present, label=None, conf=0.0):
    out = [dm.update(present, label, conf) for _ in range(n)]
    return [d for d in out if d]


def test_one_decision_per_item():
    dm = DecisionMaker(settle_frames=3, window=5, empty_frames=2)
    assert feed(dm, 3, True, "plastik", 0.9) == []
    got = feed(dm, 20, True, "plastik", 0.9)
    assert len(got) == 1 and got[0].label == "plastik" and got[0].agreement == 1.0
    assert dm.state == WAIT
    feed(dm, 2, False)
    assert dm.state == EMPTY


def test_low_confidence_is_unsure():
    dm = DecisionMaker(settle_frames=1, window=4, min_confidence=0.6)
    got = feed(dm, 10, True, "cam", 0.4)
    assert got[0].label == UNSURE


def test_split_vote_is_unsure():
    dm = DecisionMaker(settle_frames=0, window=4, min_agreement=0.6)
    dm.update(True)
    labels = ["cam", "plastik", "cam", "plastik"]
    got = [dm.update(True, l, 0.9) for l in labels]
    assert got[-1].label == UNSURE and got[-1].agreement == 0.5


def test_removed_before_decision():
    dm = DecisionMaker(settle_frames=1, window=10)
    feed(dm, 4, True, "cam", 0.9)
    assert feed(dm, 1, False) == [] and dm.state == EMPTY


def test_ledger_totals_and_replay(tmp_path):
    p = str(tmp_path / "log.csv")
    led = ClimateLedger(BINS, p)
    led.record("plastik", 0.9, 1.0)
    led.record("cam", 0.8, 0.9)
    led.record(UNSURE)
    s = led.snapshot()
    assert s["toplam"] == 3 and s["geri_kazanilan"] == 2
    assert abs(s["co2_kg"] - (0.03 + 0.09)) < 1e-9
    again = ClimateLedger(BINS, p).snapshot()
    assert again["sayilar"] == s["sayilar"] and abs(again["co2_kg"] - s["co2_kg"]) < 1e-9


def test_dataset_folder_mapping():
    cases = {"Food Organics": "organik", "Vegetation": "organik", "Cardboard": "kagit",
             "Paper": "kagit", "Plastic": "plastik", "Glass": "cam", "brown-glass": "cam",
             "Metal": "metal", "Miscellaneous Trash": "diger", "Textile Trash": "diger",
             "biological": "organik", "trash": "diger", "plastik": "plastik"}
    for name, want in cases.items():
        assert bin_of(name) == want, name
