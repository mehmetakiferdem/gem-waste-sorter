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
             "biological": "organik", "trash": "diger", "plastik": "plastik",
             "aluminium": "metal", "polystyrene": "plastik", "soft plastics": "plastik",
             "hard plastic": "plastik", "clothes": "diger", "shoes": "diger",
             "battery": None, "batteries": None, "paper towel": None,
             "takeaway cups": None, "disposable plates": None}
    for name, want in cases.items():
        assert bin_of(name) == want, name


def test_stuck_scene_asks_for_rebase():
    dm = DecisionMaker(settle_frames=1, window=3, stuck_frames=5)
    feed(dm, 10, True, "cam", 0.9)          # decides, then the scene never empties
    assert dm.needs_rebase and dm.state == EMPTY


def test_finder_skips_room_and_edge_boxes():
    import types
    from detector import ObjectFinder
    f = types.SimpleNamespace(max_area=0.7, fixture_frames=5, fixtures=[], still=(None, 0))
    pick = lambda dets: ObjectFinder.pick(f, dets, 1280, 720)
    held = ("bottle", 0.6, (500, 200, 700, 600))
    grille = ("bottle", 0.9, (870, 380, 1280, 720))      # cut off by right and bottom edges
    assert pick([grille, held]) == held
    assert pick([("microwave", 0.9, (400, 200, 800, 500))]) is None
    assert pick([("person", 0.9, (300, 0, 900, 720)), held]) == held
    assert pick([("cat", 0.9, (500, 100, 700, 300))]) is None    # ceiling lamp


def test_finder_learns_fixtures():
    import types
    from detector import ObjectFinder
    f = types.SimpleNamespace(max_area=0.7, fixture_frames=5, fixtures=[], still=(None, 0))
    pick = lambda dets: ObjectFinder.pick(f, dets, 1280, 720)
    lamp = ("vase", 0.8, (500, 100, 700, 300))
    assert [pick([lamp]) for _ in range(5)] == [lamp] * 4 + [None]
    assert pick([lamp]) is None                        # remembered
    held = ("bottle", 0.6, (480, 300, 720, 650))
    for dx in range(10):                               # a held item jitters, never fixed
        moved = (held[0], held[1], (480 + dx * 15, 300, 720 + dx * 15, 650))
        assert pick([lamp, moved]) == moved
