import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _history_row(
    variant_id="B0",
    seed=0,
    epoch=1,
    loss=1.0,
    top1=0.0,
    topk=0.0,
    train_tile_id="tile_train",
):
    return {
        "variant_id": variant_id,
        "train_tile_id": train_tile_id,
        "seed": seed,
        "epoch": epoch,
        "loss": loss,
        "top1_accuracy": top1,
        "topk_hit_rate": topk,
        "learning_rate": 0.001,
        "hidden_dim": 64,
        "claim_boundary": "phase63",
    }


def test_phase64_splits_semicolon_values_and_summarizes_convergence():
    from paper11_geofm.phase64_set_policy_error_diagnosis import (
        _split_semicolon_values,
        build_phase64_convergence_summary,
    )

    assert _split_semicolon_values(" b2 ; b1;;b3 ") == ["b2", "b1", "b3"]
    assert _split_semicolon_values("") == []

    history_rows = [
        _history_row("B0", 0, 1, loss=4.0, top1=0.0, topk=0.0),
        _history_row("B0", 0, 2, loss=2.0, top1=0.25, topk=0.50),
        _history_row("B0", 0, 3, loss=2.5, top1=0.20, topk=0.75),
        _history_row("D4P8", 1, 1, loss=3.0, top1=0.10, topk=0.25),
        _history_row("D4P8", 1, 2, loss=1.5, top1=0.40, topk=0.50),
    ]

    summary = build_phase64_convergence_summary(history_rows)

    assert len(summary) == 2
    b0 = summary[0]
    assert b0["variant_id"] == "B0"
    assert b0["seed"] == 0
    assert b0["first_epoch"] == 1
    assert b0["final_epoch"] == 3
    assert b0["best_epoch"] == 2
    assert b0["first_loss"] == 4.0
    assert b0["final_loss"] == 2.5
    assert b0["best_loss"] == 2.0
    assert b0["final_top1_accuracy"] == 0.2
    assert b0["best_top1_accuracy"] == 0.25
    assert b0["final_topk_hit_rate"] == 0.75
    assert b0["best_topk_hit_rate"] == 0.75
    assert b0["loss_delta"] == -1.5
