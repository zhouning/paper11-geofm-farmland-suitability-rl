from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_repository_files_exist():
    required_paths = [
        "README.md",
        "requirements.txt",
        "pytest.ini",
        ".gitignore",
        "reproducibility/REPRODUCTION_GUIDE.md",
        "reproducibility/DATA_MANIFEST.md",
        "reproducibility/FILE_MANIFEST.tsv",
        "scripts/smoke_check.py",
        "docs/source_notes/paper11_design_thought.md",
        "docs/superpowers/specs/2026-06-08-phase1-bishan-geofm-baseline-design.md",
        "docs/superpowers/plans/2026-06-08-phase1-bishan-geofm-baseline.md",
        "paper/design/01_design_synthesis.md",
        "paper/design/02_system_design.md",
        "paper/design/03_experiment_plan.md",
        "paper/design/04_manuscript_outline.md",
        "paper/design/05_risks_and_boundaries.md",
        "experiments/phase1_bishan_baseline/run_phase1.py",
        "experiments/geofm_runtime/embedding_space_env.py",
        "experiments/geofm_runtime/train_embedding_rl.py",
        "src/paper11_geofm/__init__.py",
        "src/paper11_geofm/sample_data.py",
        "src/paper11_geofm/regions.py",
        "src/paper11_geofm/features.py",
        "src/paper11_geofm/suitability.py",
        "src/paper11_geofm/artifacts.py",
        "src/legacy_runtime/county_env.py",
        "tests/test_phase1_geofm.py",
        "data/bishan_alphaearth_sample/metadata.json",
    ]

    missing = [path for path in required_paths if not (ROOT / path).exists()]

    assert missing == []


def test_bishan_alphaearth_sample_contains_expected_years():
    sample_dir = ROOT / "data" / "bishan_alphaearth_sample"
    expected_files = [f"bishan_emb_{year}.npy" for year in range(2017, 2025)]
    expected_files.append("bishan_context.npy")
    expected_files.append("metadata.json")

    missing = [name for name in expected_files if not (sample_dir / name).exists()]

    assert missing == []
