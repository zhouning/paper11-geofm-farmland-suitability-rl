import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_preflight_accepts_current_phase46_bundle():
    import paper11_submission_preflight as preflight

    result = preflight.run_preflight(ROOT)

    assert result["ok"] is True
    assert result["bundle_hash_ok"] is True
    assert result["content_hashes_ok"] is True
    assert result["zip_entries_ok"] is True
    assert result["claim_boundary_ok"] is True
    assert result["missing_files"] == []
    assert "Paper11_phase46_submission_bundle.zip" in result["bundle_path"]
    assert result["docx_checks"]["Paper11_formal_conclusion_manuscript.docx"] is True
    assert result["docx_checks"]["Paper11_cover_letter_and_declarations.docx"] is True


def test_preflight_reports_missing_bundle_files(tmp_path):
    import paper11_submission_preflight as preflight

    result = preflight.run_preflight(tmp_path)

    assert result["ok"] is False
    assert result["missing_files"]
    assert result["bundle_hash_ok"] is False
    assert result["content_hashes_ok"] is False


def test_preflight_cli_writes_json_report(tmp_path):
    script = ROOT / "scripts" / "paper11_submission_preflight.py"
    output = tmp_path / "preflight.json"

    completed = subprocess.run(
        [sys.executable, str(script), "--root", str(ROOT), "--json-out", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["bundle_hash_ok"] is True
    assert report["content_hashes_ok"] is True