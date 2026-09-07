from pathlib import Path

import pandas as pd

from sleep_stage_classification.data import discover_sleep_edf_pairs, sleep_edf_pair_key, trim_outer_wake_epochs


def test_sleep_edf_pair_key_uses_first_six_characters():
    assert sleep_edf_pair_key("SC4001E0-PSG.edf") == "SC4001"
    assert sleep_edf_pair_key("ST7011JP-Hypnogram.edf") == "ST7011"


def test_discover_sleep_edf_pairs_matches_psg_to_hypnogram(tmp_path: Path):
    (tmp_path / "sleep-cassette").mkdir()
    psg = tmp_path / "sleep-cassette" / "SC4001E0-PSG.edf"
    hyp = tmp_path / "sleep-cassette" / "SC4001EC-Hypnogram.edf"
    psg.write_text("", encoding="utf-8")
    hyp.write_text("", encoding="utf-8")
    pairs = discover_sleep_edf_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].record_id == "SC4001"
    assert pairs[0].psg_path == psg
    assert pairs[0].hypnogram_path == hyp


def test_trim_outer_wake_epochs_keeps_sleep_window_margin():
    frame = pd.DataFrame({"stage": ["W"] * 5 + ["N2"] * 2 + ["W"] * 5})
    trimmed = trim_outer_wake_epochs(frame, keep_minutes=1, epoch_seconds=30)
    assert trimmed["stage"].tolist() == ["W", "W", "N2", "N2", "W", "W"]
    assert trimmed["epoch_index"].tolist() == list(range(6))
