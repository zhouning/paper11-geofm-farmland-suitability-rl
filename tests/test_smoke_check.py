import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_smoke_check_validates_repository_layout():
    import smoke_check

    result = smoke_check.run_checks(ROOT)

    assert result.ok
    assert result.sample_years == list(range(2017, 2025))
    assert result.embedding_shape[-1] == 64
