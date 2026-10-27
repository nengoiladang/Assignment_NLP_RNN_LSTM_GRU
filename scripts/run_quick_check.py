"""Execute the delivered notebook on real IMDB data, in quick mode only."""
from pathlib import Path
import ast
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
RUNTIME.mkdir(exist_ok=True)
for variable, subdirectory in [
    ("JUPYTER_CONFIG_DIR", "jupyter_config"),
    ("JUPYTER_DATA_DIR", "jupyter_data"),
    ("JUPYTER_RUNTIME_DIR", "jupyter_runtime"),
    ("IPYTHONDIR", "ipython"),
    ("MPLCONFIGDIR", "matplotlib"),
]:
    directory = RUNTIME / subdirectory
    directory.mkdir(exist_ok=True)
    os.environ[variable] = str(directory)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import nbformat
from nbclient import NotebookClient
import pandas as pd


def main():
    print("Starting notebook verification...", flush=True)
    notebook = nbformat.read(ROOT / "IMDB_Sentiment_Analysis.ipynb", as_version=4)
    nbformat.validate(notebook)
    parameters = [cell for cell in notebook.cells if "parameters" in cell.metadata.get("tags", [])]
    assert len(parameters) == 1
    assert 'RUN_MODE = "full"' in parameters[0].source
    parameters[0].source = parameters[0].source.replace('RUN_MODE = "full"', 'RUN_MODE = "quick"', 1)
    # Exercise the raw-archive reader even when the real data comes from the Parquet mirror.
    archive_check = nbformat.v4.new_code_cell('''
import io, tempfile
with tempfile.TemporaryDirectory(dir=DATA_DIR) as temporary:
    fixture_path = Path(temporary) / "fixture.tar.gz"
    fixture_rows = {
        "aclImdb/train/pos/1_10.txt": "A good movie.",
        "aclImdb/test/neg/2_1.txt": "A bad movie.",
        "aclImdb/train/unsup/3_0.txt": "Must not enter the labeled dataset.",
    }
    with tarfile.open(fixture_path, "w:gz") as fixture_archive:
        for member_name, value in fixture_rows.items():
            content = value.encode("utf-8")
            member = tarfile.TarInfo(member_name)
            member.size = len(content)
            fixture_archive.addfile(member, io.BytesIO(content))
    fixture_frame = read_stanford_archive(fixture_path)
    assert len(fixture_frame) == 2
    assert set(fixture_frame.label) == {0, 1}
    assert set(fixture_frame.official_split) == {"train", "test"}
    assert not fixture_frame.text.str.contains("Must not").any()
print("Raw Stanford archive reader: PASS")
''')
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and "raw_df, provenance = load_imdb()" in cell.source:
            notebook.cells.insert(index + 1, archive_check)
            break
    else:
        raise AssertionError("Dataset loader cell not found")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    initial_report = (ROOT / "Bao_cao_IMDB.md").read_bytes()
    output = ROOT / "results" / "quick" / "IMDB_Sentiment_Analysis_quick.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)

    def announce(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"[{cell_index + 1}/{len(notebook.cells)}] {cell.source.splitlines()[0][:95]}", flush=True)

    client = NotebookClient(notebook, timeout=1800, startup_timeout=90, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}}, on_cell_start=announce)
    start = time.perf_counter()
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    assert (ROOT / "Bao_cao_IMDB.md").read_bytes() == initial_report, "Quick overwrote the full report"
    metrics = pd.read_csv(output.parent / "metrics.csv")
    assert set(metrics.model) == {"Logistic Regression", "SVM", "RNN", "LSTM"}
    assert metrics.run_mode.eq("quick").all() and metrics.test_samples.eq(1000).all()
    config = json.loads((output.parent / "config.json").read_text(encoding="utf-8"))
    assert config["split_sizes"] == {"train": 1000, "validation": 200, "test": 1000}
    assert config["max_epochs"] == 1
    assert (ROOT / "results" / "IMDB_quick_results.zip").exists()
    for name in ["data_overview", "learning_curves", "comparison", "confusion_matrices"]:
        assert (output.parent / "figures" / f"{name}.png").stat().st_size > 1000
    assert not any(out.output_type == "error" for cell in notebook.cells if cell.cell_type == "code"
                   for out in cell.get("outputs", []))
    print(metrics.to_string(index=False), flush=True)
    print(f"PASS: notebook + archive loader + four models + exports ({time.perf_counter() - start:.1f}s)", flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
