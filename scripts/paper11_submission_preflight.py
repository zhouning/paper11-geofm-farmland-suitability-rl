from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile


FINAL_DIR = Path("paper/submission/final")
BUNDLE_NAME = "Paper11_phase46_submission_bundle.zip"
BUNDLE_HASH_NAME = "Paper11_phase46_submission_bundle_sha256.txt"
CONTENT_HASH_NAME = "Paper11_phase46_submission_contents_sha256.txt"

EXPECTED_BUNDLE_ENTRIES = [
    "Paper11_cover_letter_and_declarations.docx",
    "Paper11_cover_letter_and_declarations.md",
    "Paper11_formal_conclusion_manuscript.docx",
    "Paper11_formal_conclusion_manuscript.md",
    CONTENT_HASH_NAME,
    "README.md",
]

REQUIRED_FILES = [
    BUNDLE_NAME,
    BUNDLE_HASH_NAME,
    CONTENT_HASH_NAME,
    "Paper11_formal_conclusion_manuscript.docx",
    "Paper11_formal_conclusion_manuscript.md",
    "Paper11_cover_letter_and_declarations.docx",
    "Paper11_cover_letter_and_declarations.md",
    "README.md",
]

DOCX_REQUIRED_TEXT = {
    "Paper11_formal_conclusion_manuscript.docx": [
        "Compressed GeoFM representations improve held-out farmland layout optimization",
        "compressed_geofm_route_supported",
        "compressed_route_statistically_robust",
        "cluster_directional_support",
        "cluster_magnitude_support",
        "cluster_mean_support",
        "Phase 53",
        "Phase 54",
        "artifact_lineage_consistent",
        "phase41_independent_label_inputs_missing",
        "bounded positive compressed-GeoFM representation conclusion",
    ],
    "Paper11_cover_letter_and_declarations.docx": [
        "Paper11 Cover Letter and Declarations",
        "Compressed GeoFM representations improve",
        "Declaration of Competing Interest",
        "Claim Boundary for Upload",
        "bounded positive compressed-GeoFM",
        "Phase 54",
    ],
}

CLAIM_BOUNDARY_TEXT = [
    "bounded positive compressed-GeoFM representation manuscript package",
    "It does not claim raw GeoFM B1 superiority",
    "Current conclusion: raw GeoFM state injection remains unsupported",
    "supports compressed GeoFM state routes",
    "cluster_magnitude_support",
    "cluster_mean_support",
    "artifact_lineage_consistent",
]
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _parse_sha256_lines(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        digest, sep, name = line.partition("  ")
        if not sep or not digest or not name:
            continue
        parsed[name] = digest.upper()
    return parsed


def _docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    return "".join(node.text or "" for node in root.findall(".//w:t", ns))


def _all_present(text: str, required: Iterable[str]) -> bool:
    return all(fragment in text for fragment in required)


def _check_bundle(final_dir: Path) -> tuple[bool, bool, bool, list[str]]:
    bundle_path = final_dir / BUNDLE_NAME
    external_hash_path = final_dir / BUNDLE_HASH_NAME
    if not bundle_path.exists() or not external_hash_path.exists():
        return False, False, False, []

    try:
        with ZipFile(bundle_path) as archive:
            entries = sorted(archive.namelist())
            zip_entries_ok = entries == EXPECTED_BUNDLE_ENTRIES
            if CONTENT_HASH_NAME not in entries:
                return zip_entries_ok, False, False, entries
            expected_hashes = _parse_sha256_lines(
                archive.read(CONTENT_HASH_NAME).decode("utf-8")
            )
            content_hashes_ok = True
            for entry_name in EXPECTED_BUNDLE_ENTRIES:
                if entry_name == CONTENT_HASH_NAME:
                    continue
                if entry_name not in expected_hashes or entry_name not in entries:
                    content_hashes_ok = False
                    continue
                actual = _sha256_bytes(archive.read(entry_name))
                if expected_hashes[entry_name] != actual:
                    content_hashes_ok = False
    except (BadZipFile, KeyError, UnicodeDecodeError):
        return False, False, False, []

    bundle_digest = _sha256_file(bundle_path)
    declared_hashes = _parse_sha256_lines(external_hash_path.read_text(encoding="utf-8"))
    bundle_hash_ok = declared_hashes.get(BUNDLE_NAME) == bundle_digest
    return zip_entries_ok, content_hashes_ok, bundle_hash_ok, entries


def run_preflight(root: Path | str) -> dict[str, object]:
    root_path = Path(root)
    final_dir = root_path / FINAL_DIR
    missing_files = [
        str(FINAL_DIR / name).replace("\\", "/")
        for name in REQUIRED_FILES
        if not (final_dir / name).exists()
    ]

    zip_entries_ok, content_hashes_ok, bundle_hash_ok, bundle_entries = _check_bundle(final_dir)

    docx_checks: dict[str, bool] = {}
    for name, required_text in DOCX_REQUIRED_TEXT.items():
        path = final_dir / name
        if not path.exists():
            docx_checks[name] = False
            continue
        try:
            docx_checks[name] = _all_present(_docx_text(path), required_text)
        except (BadZipFile, KeyError, ET.ParseError):
            docx_checks[name] = False

    readme_path = final_dir / "README.md"
    claim_boundary_ok = False
    if readme_path.exists():
        claim_boundary_ok = _all_present(
            readme_path.read_text(encoding="utf-8"), CLAIM_BOUNDARY_TEXT
        )

    ok = (
        not missing_files
        and zip_entries_ok
        and content_hashes_ok
        and bundle_hash_ok
        and all(docx_checks.values())
        and claim_boundary_ok
    )

    return {
        "ok": ok,
        "bundle_path": str(FINAL_DIR / BUNDLE_NAME).replace("\\", "/"),
        "missing_files": missing_files,
        "zip_entries_ok": zip_entries_ok,
        "bundle_entries": bundle_entries,
        "content_hashes_ok": content_hashes_ok,
        "bundle_hash_ok": bundle_hash_ok,
        "docx_checks": docx_checks,
        "claim_boundary_ok": claim_boundary_ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight Paper11 submission bundle.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--json-out", help="Optional JSON report output path.")
    args = parser.parse_args(argv)

    report = run_preflight(Path(args.root))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())