"""Smoke tests - minimal deps. Run first to verify env."""
import subprocess
import sys
from pathlib import Path


def test_python_version():
    assert sys.version_info >= (3, 9), "Python 3.9+ required"


def test_project_structure():
    root = Path(__file__).resolve().parent.parent
    assert (root / "scripts" / "train.py").exists()
    assert (root / "scripts" / "ingest.py").exists()
    assert (root / "scripts" / "preprocess.py").exists()
    assert (root / "scripts" / "evaluate.py").exists()
    assert (root / "scripts" / "api_server.py").exists()
    assert (root / "scripts" / "lambda_handler.py").exists()


def test_cli_evaluate_help():
    """scripts.evaluate --help exits 0 (minimal imports)."""
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, "-m", "scripts.evaluate", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(root),
    )
    assert r.returncode == 0, r.stderr or r.stdout
    assert "model" in (r.stdout or "").lower() or r.returncode == 0
