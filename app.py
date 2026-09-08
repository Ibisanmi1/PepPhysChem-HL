from __future__ import annotations
import contextlib
import html
import inspect
import io
import math
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GITHUB_REPO = "https://github.com/Ibisanmi1/PepPhysChem-HL"
SOFTWARE_NAME = "PepPhysChem-HL"
SOFTWARE_FULL_TITLE = (
    "PepPhysChem-HL: An Integrated Command-Line and Web Platform for "
    "Physicochemical Profiling and Deep Learning-Based Half-Life Prediction "
    "of Therapeutic Peptides"
)
CITATION_BIB = PROJECT_ROOT / "CITATION.bib"
CITATION_CFF = PROJECT_ROOT / "CITATION.cff"

CITATION_INTRO = "If this pipeline contributes to your research, please cite:"
CITATION_AUTHORS = (
    "Tope Abraham Ibisanmi, Ghayah Bahatheg, Shyam Kumar Mishra (Baishnab), "
    "Mark Willcox, and Naresh Kumar"
)
CITATION_LINE = (
    f"{CITATION_AUTHORS} (2026). "
    f"{SOFTWARE_FULL_TITLE}."
)
CITATION_FULL_TEXT = (
    f"{CITATION_INTRO}\n\n{CITATION_LINE}\nAvailable from: {GITHUB_REPO}\n"
)

DEFAULT_HYBRID_CHECKPOINT_NAME = "Half_Life_cnn_bilstm_embedding_physchem_run1.pt"
DEFAULT_HYBRID_TRAINING_CONFIG = (
    PROJECT_ROOT
    / "training_logs"
    / "1_cnn_bilstm_hybrid_physchem_matrix"
    / "training_config.json"
)
DEFAULT_CNN_PHYSCHEM_CHECKPOINT_NAME = "Half_Life_cnn_embedding_physchem_run1.pt"
DEFAULT_CNN_PHYSCHEM_TRAINING_CONFIG = (
    PROJECT_ROOT
    / "training_logs"
    / "1_cnn_embedding_hybrid_physchem_matrix"
    / "training_config.json"
)

# Default: CNN–BiLSTM + physchem (embedding), same weights as before.
PRESET_CNN_BILSTM_PHYSCHEM = "cnn_bilstm_hybrid_physchem"
PRESET_CNN_PHYSCHEM = "cnn_embedding_hybrid_physchem"
# Back-compat alias used by older UI / env wiring.
PRESET_RECOMMENDED = PRESET_CNN_BILSTM_PHYSCHEM

MODEL_PRESETS: List[Tuple[str, str]] = [
    (
        "CNN–BiLSTM + physchem (emb.) — default",
        PRESET_CNN_BILSTM_PHYSCHEM,
    ),
    (
        "CNN + physchem (emb.)",
        PRESET_CNN_PHYSCHEM,
    ),
]
PRESET_DROPDOWN_LABELS = [pair[0] for pair in MODEL_PRESETS]
LABEL_TO_PRESET: Dict[str, str] = {label: key for label, key in MODEL_PRESETS}

_predictor_cache: Dict[str, Any] = {}

_FIGURE_SUFFIXES = (
    "_distributions.png",
    "_correlations.png",
    "_half_life_relationships.png",
    "_hydrophobic_analysis.png",
    "_charge_amphipathicity.png",
    "_structural_analysis.png",
)

_FIGURE_CAPTIONS: Dict[str, str] = {
    "_distributions.png": "Distributions & KDE — key peptide properties",
    "_correlations.png": "Correlations with half-life (Pearson r)",
    "_half_life_relationships.png": "Scatter & linear fits vs half-life",
    "_hydrophobic_analysis.png": "Hydrophobic descriptors vs half-life",
    "_charge_amphipathicity.png": "Charge, amphipathicity & joint views",
    "_structural_analysis.png": "Structural descriptors vs half-life",
}

_DEFAULT_WEB_SAVE_DPI = int(
    os.environ.get("PEPPHYSCHEM_HL_WEB_FIGURE_DPI")
    or os.environ.get("AMP_WEB_FIGURE_DPI")
    or "420"
)


@contextlib.contextmanager
def _high_res_matplotlib_saves(dpi: Optional[int] = None) -> Any:
    """Temporarily force higher savefig DPI for publication-quality PNGs in the UI."""
    import matplotlib.pyplot as plt

    target = dpi if dpi is not None else _DEFAULT_WEB_SAVE_DPI
    orig = plt.savefig

    def _savefig(*args: Any, **kwargs: Any) -> Any:
        kw = dict(kwargs)
        kw["dpi"] = max(int(kw.get("dpi") or 0), int(target))
        return orig(*args, **kw)

    plt.savefig = _savefig
    try:
        yield
    finally:
        plt.savefig = orig


def _prepare_results_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Remove error-string column so it does not interfere with numeric summaries."""
    out = df.copy()
    if "physchem_error" in out.columns:
        out = out.drop(columns=["physchem_error"], errors="ignore")
    return out


def _captioned_gallery(paths: List[str]) -> List[Tuple[str, str]]:
    """(path, caption) tuples for Gradio Gallery."""
    items: List[Tuple[str, str]] = []
    for p in paths:
        name = Path(p).name
        cap = next(
            (lbl for suf, lbl in _FIGURE_CAPTIONS.items() if name.endswith(suf)),
            name,
        )
        items.append((p, cap))
    return items


def _zip_figure_bundle(paths: List[str], prefix: str) -> Optional[str]:
    if not paths:
        return None
    out = _gradio_figure_dir() / f"{prefix}_figures_hires.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            fp = Path(p)
            if fp.is_file():
                zf.write(fp, arcname=fp.name)
    return str(out.resolve())


def _gradio_figure_dir() -> Path:
    d = PROJECT_ROOT / "output" / "gradio_figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _collect_gradio_figures(out_dir: Path, prefix: str) -> List[str]:
    paths: List[str] = []
    for suf in _FIGURE_SUFFIXES:
        p = (out_dir / f"{prefix}{suf}").resolve()
        if p.is_file():
            paths.append(str(p))
    return paths


def _run_comprehensive_figures(results_df: pd.DataFrame, prefix: str) -> Tuple[List[str], str]:
    """Run publication-style analysis; return absolute paths to PNGs and a short markdown note."""
    import run_PepPhysChem_HL as runner

    out_dir = _gradio_figure_dir()
    clean_df = _prepare_results_for_analysis(results_df)
    try:
        with contextlib.redirect_stdout(io.StringIO()), _high_res_matplotlib_saves():
            comp = runner.ComprehensiveAnalysis(out_dir)
            comp.analyze_results(clean_df, output_prefix=prefix)
    except Exception as e:
        return [], f"### Analysis figures\n*Generation failed:* `{type(e).__name__}`: {e}"

    imgs = _collect_gradio_figures(out_dir, prefix)
    if not imgs:
        return [], (
            "### Analysis figures\n"
            "*No figure PNGs were written.* With very few sequences, correlation and scatter plots "
            "are skipped by the pipeline. Use **Batch** with **≥4 sequences** (and physicochemical "
            "profile enabled) for the full figure set. Distributions may still appear for a single row."
        )
    dpi_note = _DEFAULT_WEB_SAVE_DPI
    return imgs, (
        f"### Publication figures\n**{len(imgs)}** high-resolution panel(s) "
        f"(**{dpi_note} dpi** PNG). Click any thumbnail for **fullscreen preview**. "
        f"Files: `output/gradio_figures/{prefix}_*.png`. Use **Download figure bundle (ZIP)** for originals."
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return f"(File not found: {path.name})"


def _build_citation_zip() -> str:
    """Bundle BibTeX, CITATION.cff, and plain-text CITATION.txt (GitHub-style pack)."""
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "PepPhysChem-HL_citation.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if CITATION_BIB.is_file():
            zf.write(CITATION_BIB, arcname="CITATION.bib")
        if CITATION_CFF.is_file():
            zf.write(CITATION_CFF, arcname="CITATION.cff")
        zf.writestr("CITATION.txt", CITATION_FULL_TEXT.strip() + "\n")
    return str(zip_path)


def _default_hybrid_checkpoint_path() -> Path:
    return PROJECT_ROOT / "checkpoints" / DEFAULT_HYBRID_CHECKPOINT_NAME


def _checkpoint_missing_message(exc: BaseException) -> str:
    expected = _default_hybrid_checkpoint_path()
    return (
        f"### Model checkpoint not found\n\n{exc}\n\n"
        f"Place **`{DEFAULT_HYBRID_CHECKPOINT_NAME}`** under `checkpoints/` "
        f"(expected: `{expected}`) or set **`PEPPHYSCHEM_HL_MODEL_PATH`** to your `.pt` file."
    )


def _checkpoint_load_error_message(exc: BaseException) -> str:
    hint = ""
    if "invalid load key" in str(exc).lower() and "'v'" in str(exc):
        hint = (
            "\n\nThis usually means the Space checked out a **Git LFS pointer** "
            "instead of the real `.pt` file. Redeploy with the latest `app.py` "
            "(which downloads weights via the Hub), or upload the checkpoint via "
            "**Files → Add file** on the Space."
        )
    return f"### Model checkpoint could not be loaded\n\n`{type(exc).__name__}`: {exc}{hint}"


def _checkpoint_roots() -> List[Path]:
    import run_PepPhysChem_HL as runner

    roots = [PROJECT_ROOT]
    ai = runner.PEPPHYSOCHEM_HL_AI_ROOT
    if ai.is_dir() and ai.resolve() != PROJECT_ROOT.resolve():
        roots.append(ai)
    return roots


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _is_pytorch_checkpoint_file(path: Path) -> bool:
    """True when the file looks like a real torch.save artifact (not an LFS stub)."""
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError:
        return False
    if head[:2] == b"PK":
        return True
    return bool(head) and head[0:1] in (b"\x80", b"\x85", b"\x86", b"\x87", b"\x88")


def _hf_space_repo_id() -> str:
    return (
        os.environ.get("PEPPHYSCHEM_HL_HF_SPACE_ID")
        or os.environ.get("SPACE_ID")
        or "Ibisanmi1/PepPhysChem-HL"
    ).strip()


def _download_checkpoint_from_hf_space(basename: str) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=_hf_space_repo_id(),
        filename=f"checkpoints/{basename}",
        repo_type="space",
    )
    return Path(downloaded).resolve()


def _materialize_checkpoint(path: Path) -> Path:
    """
    Return a loadable checkpoint path.

    On Hugging Face Spaces, Git LFS pointer stubs are checked out as small text
    files starting with ``version https://git-lfs...``; torch.load then fails with
    ``UnpicklingError: invalid load key, 'v'``. Resolve those via the Hub API.
    """
    resolved = path.resolve()
    if _is_pytorch_checkpoint_file(resolved):
        return resolved
    if _is_git_lfs_pointer(resolved):
        return _download_checkpoint_from_hf_space(resolved.name)
    raise ValueError(
        f"Checkpoint at `{resolved}` is not a valid PyTorch file. "
        "Upload the real `.pt` weights or set PEPPHYSOCHEM_HL_HF_SPACE_ID."
    )


def _find_checkpoint(basenames: List[str]) -> Optional[Path]:
    for root in _checkpoint_roots():
        ck = root / "checkpoints"
        for name in basenames:
            p = ck / name
            if p.is_file():
                return _materialize_checkpoint(p)
    return None


def _resolve_preset(preset_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (model_path, training_config_path) for PepPhysChemHLPredictor.
    """
    env_mp = (
        os.environ.get("PEPPHYSCHEM_HL_MODEL_PATH")
        or os.environ.get("AMP_MODEL_PATH")
        or ""
    ).strip() or None
    expected_bilstm = _default_hybrid_checkpoint_path()
    expected_cnn = PROJECT_ROOT / "checkpoints" / DEFAULT_CNN_PHYSCHEM_CHECKPOINT_NAME

    if preset_key in (PRESET_CNN_BILSTM_PHYSCHEM, PRESET_RECOMMENDED, "__recommended__"):
        if env_mp:
            return str(_materialize_checkpoint(Path(env_mp))), None
        ck = _find_checkpoint([DEFAULT_HYBRID_CHECKPOINT_NAME])
        if ck is None:
            raise FileNotFoundError(
                "Checkpoint for CNN–BiLSTM + physchem (emb.) not found. "
                f"Expected `{expected_bilstm}` "
                "(or the same filename under PEPPHYSOCHEM_HL_AI_ROOT/checkpoints/)."
            )
        tcp = (
            str(DEFAULT_HYBRID_TRAINING_CONFIG)
            if DEFAULT_HYBRID_TRAINING_CONFIG.is_file()
            else None
        )
        return str(ck), tcp

    if preset_key == PRESET_CNN_PHYSCHEM:
        ck = _find_checkpoint(
            [
                DEFAULT_CNN_PHYSCHEM_CHECKPOINT_NAME,
                "Half_Life_cnn_embedding_physchem.pt",
            ]
        )
        if ck is None:
            raise FileNotFoundError(
                "Checkpoint for CNN + physchem (emb.) not found. "
                f"Expected `{expected_cnn}` "
                "(or the same filename under PEPPHYSOCHEM_HL_AI_ROOT/checkpoints/)."
            )
        tcp = (
            str(DEFAULT_CNN_PHYSCHEM_TRAINING_CONFIG)
            if DEFAULT_CNN_PHYSCHEM_TRAINING_CONFIG.is_file()
            else None
        )
        return str(ck), tcp

    raise ValueError(f"Unknown model preset: {preset_key}")


def _get_predictor(preset_key: str) -> Any:
    if preset_key not in _predictor_cache:
        import run_PepPhysChem_HL as runner

        mp, tcp = _resolve_preset(preset_key)
        with contextlib.redirect_stdout(io.StringIO()):
            _predictor_cache[preset_key] = runner.PepPhysChemHLPredictor(
                model_path=mp,
                device=(
                    os.environ.get("PEPPHYSCHEM_HL_DEVICE")
                    or os.environ.get("AMP_DEVICE")
                    or None
                ),
                training_config_path=tcp,
            )
    return _predictor_cache[preset_key]


_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _clean_standard_aa_string(s: str) -> str:
    return "".join(c for c in (s or "").upper().strip() if c in _STANDARD_AA)


def _aa_composition_chart_png(
    seq_clean: str,
    *,
    title: str,
    xlabel: str,
    filename_prefix: str,
) -> Optional[str]:
    """
    Horizontal bar chart: % each standard AA with count > 0.
    Error bars: ±1 SE (multinomial), sqrt(p̂(1−p̂)/n) × 100.
    """
    if not seq_clean:
        return None
    counts = Counter(seq_clean)
    items = [(aa, c) for aa, c in sorted(counts.items()) if c > 0]
    if not items:
        return None
    n = len(seq_clean)
    labels = [x[0] for x in items]
    pcts = [100.0 * x[1] / n for x in items]
    se_pct: List[float] = []
    for _aa, cnt in items:
        ph = cnt / n
        if ph <= 0.0 or ph >= 1.0:
            se_pct.append(0.0)
        else:
            se_pct.append(100.0 * math.sqrt(ph * (1.0 - ph) / n))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import run_PepPhysChem_HL as runner

    primary = runner.CHART_PRIMARY
    edge = runner.CHART_EDGE

    fig_h = max(3.2, 0.38 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_h), layout="constrained")
    y_pos = range(len(labels))
    ax.barh(
        list(y_pos),
        pcts,
        color=primary,
        edgecolor=edge,
        linewidth=0.8,
        xerr=se_pct,
        capsize=3,
        error_kw={
            "ecolor": edge,
            "elinewidth": 1.0,
            "capthick": 1.0,
            "alpha": 0.9,
        },
    )
    ax.set_yticks(list(y_pos), labels=labels, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.text(
        0.99,
        0.02,
        "Error bars: ±1 SE (multinomial proportion)",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        color="#4a5568",
        style="italic",
    )
    max_x = max((p + s for p, s in zip(pcts, se_pct)), default=0.0)
    ax.set_xlim(0, min(100.0, max_x * 1.18 + 1.5) if max_x else 1.0)
    ax.grid(True, axis="x", alpha=0.35, linestyle="--")
    for i, (p, s) in enumerate(zip(pcts, se_pct)):
        ax.text(p + s + 0.25, i, f"{p:.1f}%", va="center", fontsize=10, fontweight="bold")
    out = _gradio_figure_dir() / f"{filename_prefix}_{int(time.time())}.png"
    dpi = min(int(_DEFAULT_WEB_SAVE_DPI), 300)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return str(out.resolve())


def _single_sequence_aa_composition_chart_png(sequence: str) -> Optional[str]:
    seq_clean = _clean_standard_aa_string(sequence)
    if not seq_clean:
        return None
    n = len(seq_clean)
    return _aa_composition_chart_png(
        seq_clean,
        title=f"Amino acid composition — present residues only (n = {n} standard AAs)",
        xlabel="Percentage of sequence (%)",
        filename_prefix="gradio_single_aa_composition",
    )


def _batch_pooled_aa_composition_chart_png(sequences: List[str]) -> Optional[str]:
    """Pooled standard-AA string over all batch rows (same order as input / results table)."""
    pooled = "".join(_clean_standard_aa_string(s) for s in sequences)
    if not pooled:
        return None
    k = len(sequences)
    n = len(pooled)
    return _aa_composition_chart_png(
        pooled,
        title=f"Amino acid composition — pooled batch ({k} sequences, n = {n} standard AA residues)",
        xlabel="Percentage of pooled residues (%)",
        filename_prefix="gradio_batch_aa_composition",
    )


def _format_single_markdown(result: Dict[str, Any]) -> str:
    seq = result.get("sequence", "")
    ln = result.get("length", len(str(seq)))
    hl = float(result.get("half_life", 0))
    lines = [
        "### Predicted half-life",
        f"**{hl:.2f} min** · *{hl / 60.0:.2f} h*",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Sequence | `{seq}` |",
        f"| Length (aa) | {ln} |",
    ]
    err = result.get("physchem_error")
    if err:
        lines.extend(["", f"*Physicochemical extension note: {err}*"])
    return "\n".join(lines)


def _result_to_table(result: Dict[str, Any]) -> pd.DataFrame:
    skip = {"sequence", "half_life", "length", "physchem_error"}
    rows: List[Tuple[str, str]] = []
    for key in sorted(result.keys()):
        if key in skip:
            continue
        val = result[key]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            rows.append((key, "—"))
        elif isinstance(val, float):
            rows.append((key, f"{val:.6g}"))
        else:
            rows.append((key, str(val)))
    if not rows:
        return pd.DataFrame(columns=["Property", "Value"])
    return pd.DataFrame(rows, columns=["Property", "Value"])


def predict_single(
    model_label: str,
    sequence: str,
    include_physchem: bool,
    gen_figures: bool,
) -> Tuple[str, pd.DataFrame, Optional[str], str, List[Tuple[str, str]], Optional[str]]:
    seq = (sequence or "").strip().upper().replace(" ", "").replace("\n", "")
    preset = LABEL_TO_PRESET.get(model_label, PRESET_RECOMMENDED)
    if not seq:
        return (
            "### Input required\nEnter a one-letter amino acid sequence.",
            pd.DataFrame(),
            None,
            "",
            [],
            None,
        )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            p = _get_predictor(preset)
            result = p.analyze_single(seq, include_physchem=include_physchem)
    except FileNotFoundError as e:
        return (
            _checkpoint_missing_message(e),
            pd.DataFrame(),
            None,
            "",
            [],
            None,
        )
    except AttributeError as e:
        if "DEFAULT_HYBRID_CHECKPOINT_NAME" in str(e) or "default_hybrid_checkpoint_path" in str(e):
            return (
                _checkpoint_missing_message(
                    FileNotFoundError(
                        f"Deploy `run_PepPhysChem_HL.py` is out of date ({e}). "
                        f"Expected checkpoint: `{_default_hybrid_checkpoint_path()}`"
                    )
                ),
                pd.DataFrame(),
                None,
                "",
                [],
                None,
            )
        return (f"### Error\n\n`{type(e).__name__}`: {e}", pd.DataFrame(), None, "", [], None)
    except Exception as e:
        if e.__class__.__name__ == "UnpicklingError" or "invalid load key" in str(e).lower():
            return (
                _checkpoint_load_error_message(e),
                pd.DataFrame(),
                None,
                "",
                [],
                None,
            )
        return (f"### Error\n\n`{type(e).__name__}`: {e}", pd.DataFrame(), None, "", [], None)

    md = _format_single_markdown(result)
    df = _result_to_table(result) if include_physchem else pd.DataFrame()

    aa_chart: Optional[str] = None
    try:
        aa_chart = _single_sequence_aa_composition_chart_png(result.get("sequence", seq))
    except Exception:
        aa_chart = None

    fig_md = ""
    gallery: List[Tuple[str, str]] = []
    zip_path: Optional[str] = None
    if gen_figures and "half_life" in result:
        wide = pd.DataFrame([result])
        prefix = f"gradio_single_{int(time.time())}"
        imgs, fig_md = _run_comprehensive_figures(wide, prefix)
        gallery = _captioned_gallery(imgs)
        zip_path = _zip_figure_bundle(imgs, prefix)

    return md, df, aa_chart, fig_md, gallery, zip_path


def predict_batch(
    model_label: str,
    text_block: str,
    include_physchem: bool,
    gen_figures: bool,
) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str], str, List[Tuple[str, str]], Optional[str]]:
    raw = (text_block or "").strip().splitlines()
    sequences = []
    for line in raw:
        s = line.strip().upper().replace(" ", "")
        if not s or s.startswith("#"):
            continue
        sequences.append(s)
    sequences = sequences[:200]
    preset = LABEL_TO_PRESET.get(model_label, PRESET_RECOMMENDED)
    if not sequences:
        return (
            pd.DataFrame(),
            "### Batch\nEnter one sequence per line (max 200).",
            None,
            None,
            "",
            [],
            None,
        )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            p = _get_predictor(preset)
            df = p.analyze_batch(sequences, include_physchem=include_physchem, progress=False)
    except FileNotFoundError as e:
        return (
            pd.DataFrame(),
            _checkpoint_missing_message(e),
            None,
            None,
            "",
            [],
            None,
        )
    except AttributeError as e:
        if "DEFAULT_HYBRID_CHECKPOINT_NAME" in str(e) or "default_hybrid_checkpoint_path" in str(e):
            return (
                pd.DataFrame(),
                _checkpoint_missing_message(
                    FileNotFoundError(
                        f"Deploy `run_PepPhysChem_HL.py` is out of date ({e}). "
                        f"Expected checkpoint: `{_default_hybrid_checkpoint_path()}`"
                    )
                ),
                None,
                None,
                "",
                [],
                None,
            )
    except Exception as e:
        if e.__class__.__name__ == "UnpicklingError" or "invalid load key" in str(e).lower():
            return (
                pd.DataFrame(),
                _checkpoint_load_error_message(e),
                None,
                None,
                "",
                [],
                None,
            )
        return (
            pd.DataFrame(),
            f"### Error\n\n`{type(e).__name__}`: {e}",
            None,
            None,
            "",
            [],
            None,
        )

    csv_path = None
    try:
        out_dir = PROJECT_ROOT / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = str(out_dir / "gradio_batch_last.csv")
        df.to_csv(csv_path, index=False)
    except OSError:
        csv_path = None

    summary = (
        f"### Batch complete\n**{len(df)}** sequences · "
        f"half-life mean **{df['half_life'].mean():.2f}** min "
        f"(min {df['half_life'].min():.2f}, max {df['half_life'].max():.2f})"
    )

    aa_batch_chart: Optional[str] = None
    try:
        seqs_for_aa = (
            df["sequence"].astype(str).tolist()
            if not df.empty and "sequence" in df.columns
            else sequences
        )
        aa_batch_chart = _batch_pooled_aa_composition_chart_png(seqs_for_aa)
    except Exception:
        aa_batch_chart = None

    fig_md = ""
    gallery: List[Tuple[str, str]] = []
    zip_path: Optional[str] = None
    if gen_figures and not df.empty and "half_life" in df.columns:
        prefix = f"gradio_batch_{int(time.time())}"
        imgs, fig_md = _run_comprehensive_figures(df, prefix)
        gallery = _captioned_gallery(imgs)
        zip_path = _zip_figure_bundle(imgs, prefix)

    return df, summary, csv_path, aa_batch_chart, fig_md, gallery, zip_path


CUSTOM_CSS = """
:root {
  --unsw-navy: #002664;
  --unsw-navy-mid: #003d7a;
  --hero-blue-deep: #0d5a8c;
  --hero-accent-light: #a8d9ff;
  --hero-accent-mid: #5eb0e8;
  --ink: #1a2332;
  --muted: #5c6b7a;
  --surface: #ffffff;
  --line: #d8e2ef;
}
.gradio-container {
  max-width: 1280px !important;
  margin: auto !important;
  font-family: "Source Sans 3", "Segoe UI", system-ui, sans-serif !important;
  color: var(--ink);
  background: linear-gradient(165deg, #e8eef7 0%, #f4f7fb 35%, #fafbfc 100%) !important;
  min-height: 100vh;
  padding-bottom: 2rem !important;
}
.app-shell {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1.35rem 1.5rem 1.75rem;
  box-shadow: 0 4px 32px rgba(15, 23, 42, 0.07);
  margin-top: 0.25rem;
}
.section-heading {
  margin: 1rem 0 0.45rem 0;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--unsw-navy);
  border-bottom: 2px solid var(--hero-blue-deep);
  padding-bottom: 0.25rem;
  display: inline-block;
}
.panel-inset {
  border-radius: 12px;
  border: 1px solid #e5edf6;
  background: #fbfcfe;
  padding: 0.5rem 0.75rem;
}
.hero {
  background: linear-gradient(122deg, #001a3d 0%, var(--unsw-navy) 42%, var(--unsw-navy-mid) 72%, #0d5a8c 100%);
  color: #fff !important;
  padding: 0;
  border-radius: 12px;
  margin-bottom: 1.35rem;
  box-shadow: 0 12px 40px rgba(0, 26, 61, 0.22);
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
}
.hero-inner {
  display: grid;
  grid-template-columns: 1fr minmax(220px, 280px);
  gap: 1.5rem;
  padding: 1.85rem 2rem 1.5rem 2rem;
  align-items: start;
}
@media (max-width: 768px) {
  .hero-inner { grid-template-columns: 1fr; }
}
.hero-kicker {
  margin: 0 0 0.5rem 0;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  opacity: 0.95;
  color: #fff !important;
}
.hero h1 {
  margin: 0 0 0.5rem 0;
  font-size: clamp(1.45rem, 4vw, 1.85rem);
  font-weight: 700;
  letter-spacing: 0.015em;
  line-height: 1.2;
  color: #fff !important;
}
.hero .sub {
  margin: 0 0 1rem 0;
  opacity: 0.93;
  font-size: 1.03rem;
  line-height: 1.55;
  max-width: 52rem;
  color: #fff !important;
}
.hero .sub strong {
  color: #fff !important;
}
.hero-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem 0.65rem;
  margin-bottom: 0.25rem;
}
.hero-link {
  color: #fff !important;
  font-weight: 600;
  font-size: 0.95rem;
  text-decoration: underline !important;
  text-underline-offset: 3px;
  text-decoration-color: rgba(168, 217, 255, 0.65) !important;
}
.hero-link:hover { text-decoration-color: var(--hero-accent-light) !important; }
.hero-dot { opacity: 0.5; user-select: none; }
.hero-aside .hero-card {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 10px;
  padding: 1rem 1.1rem;
  backdrop-filter: blur(8px);
}
.hero-card-title {
  display: block;
  font-size: 0.65rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  opacity: 0.85;
  margin-bottom: 0.45rem;
  color: var(--hero-accent-light);
}
.hero-aside .hero-card p,
.hero-aside .hero-card p.hero-card-model-hint {
  margin: 0;
  font-size: 0.88rem;
  line-height: 1.45;
  opacity: 1;
  color: #ffffff !important;
}
.hero-aside .hero-card p strong {
  color: #ffffff !important;
}
.hero-aside .hero-card code {
  color: #f0f7ff !important;
  background: rgba(0,0,0,0.28) !important;
  padding: 0.12em 0.35em;
  border-radius: 4px;
  font-size: 0.9em !important;
}
div.foot {
  margin-top: 2rem;
  padding-top: 1.15rem;
  border-top: 1px solid #dde3ea;
  font-size: 0.86rem;
  color: var(--muted);
  line-height: 1.5;
}
div.foot a { color: var(--unsw-navy-mid); font-weight: 600; }
.foot-cite {
  margin-top: 1.35rem;
  padding-top: 1.15rem;
  border-top: 1px solid #dde3ea;
  font-size: 0.88rem;
  line-height: 1.55;
  color: var(--ink);
}
.foot-cite-intro {
  margin: 0 0 0.5rem 0;
  font-weight: 700;
  color: var(--unsw-navy);
}
.foot-cite-body {
  margin: 0;
  color: var(--ink);
}
.cite-panel {
  background: #fafbfd;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 1rem 1.15rem;
  margin-bottom: 0.75rem;
  color: var(--ink);
}
.cite-panel code,
.cite-panel-code {
  display: block;
  white-space: pre-wrap;
  font-size: 0.92em;
  color: var(--ink) !important;
  background: #f1f5f9 !important;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0.75rem 0.9rem;
  line-height: 1.55;
}
.figure-gallery-wrap {
  margin-top: 0.5rem;
}
/* Full-width gallery: large previews, crisp downscaled display */
.pro-gallery .grid-wrap {
  grid-template-columns: 1fr !important;
  gap: 1.25rem !important;
}
.pro-gallery img, .pro-gallery video {
  border-radius: 10px !important;
  box-shadow: 0 8px 28px rgba(0, 38, 100, 0.12) !important;
  border: 1px solid rgba(0, 38, 100, 0.08) !important;
}
.pro-gallery .thumbnail-item {
  min-height: 420px !important;
}
footer { opacity: 0.85; font-size: 0.8rem; }
.aa-composition-chart img {
  border-radius: 10px;
  border: 1px solid #e5edf6;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
  max-width: 100%;
}

/* Dark mode / night theme: keep custom panels readable (Gradio may flip global text to light). */
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e8eef7;
    --muted: #b8c5d6;
    --surface: #1a2332;
    --line: #3d4f66;
  }
  .gradio-container {
    background: linear-gradient(165deg, #0f1724 0%, #141e2e 35%, #1a2332 100%) !important;
    color: var(--ink) !important;
  }
  .app-shell {
    background: var(--surface);
    border-color: var(--line);
    box-shadow: 0 4px 32px rgba(0, 0, 0, 0.35);
  }
  .panel-inset,
  .cite-panel {
    background: #243044;
    border-color: #3d4f66;
    color: var(--ink) !important;
  }
  .cite-panel strong,
  .cite-panel code,
  .cite-panel-code {
    color: #f0f4fa !important;
    background: #1a2332 !important;
    border-color: #3d4f66;
  }
  .section-heading {
    color: #a8d9ff;
    border-bottom-color: #5eb0e8;
  }
  div.foot,
  .foot-cite {
    border-top-color: #3d4f66;
    color: var(--muted);
  }
  .foot-cite-intro {
    color: #a8d9ff;
  }
  .foot-cite-body {
    color: var(--ink);
  }
  div.foot a {
    color: #7ec8ff;
  }
  .aa-composition-chart img {
    border-color: #3d4f66;
  }
}

.dark .gradio-container,
[data-theme="dark"] .gradio-container,
.gradio-container.dark {
  background: linear-gradient(165deg, #0f1724 0%, #141e2e 35%, #1a2332 100%) !important;
  color: #e8eef7 !important;
}
.dark .app-shell,
[data-theme="dark"] .app-shell,
.gradio-container.dark .app-shell {
  background: #1a2332 !important;
  border-color: #3d4f66 !important;
  color: #e8eef7 !important;
}
.dark .panel-inset,
.dark .cite-panel,
[data-theme="dark"] .panel-inset,
[data-theme="dark"] .cite-panel,
.gradio-container.dark .panel-inset,
.gradio-container.dark .cite-panel {
  background: #243044 !important;
  border-color: #3d4f66 !important;
  color: #e8eef7 !important;
}
.dark .cite-panel code,
.dark .cite-panel-code,
[data-theme="dark"] .cite-panel code,
[data-theme="dark"] .cite-panel-code,
.gradio-container.dark .cite-panel code,
.gradio-container.dark .cite-panel-code {
  color: #f0f4fa !important;
  background: #1a2332 !important;
  border-color: #3d4f66 !important;
}
.dark .section-heading,
[data-theme="dark"] .section-heading,
.gradio-container.dark .section-heading {
  color: #a8d9ff !important;
  border-bottom-color: #5eb0e8 !important;
}
.dark .foot-cite,
.dark div.foot,
[data-theme="dark"] .foot-cite,
[data-theme="dark"] div.foot,
.gradio-container.dark .foot-cite,
.gradio-container.dark div.foot {
  border-top-color: #3d4f66 !important;
  color: #b8c5d6 !important;
}
.dark .foot-cite-intro,
[data-theme="dark"] .foot-cite-intro,
.gradio-container.dark .foot-cite-intro {
  color: #a8d9ff !important;
}
.dark .foot-cite-body,
[data-theme="dark"] .foot-cite-body,
.gradio-container.dark .foot-cite-body {
  color: #e8eef7 !important;
}
.dark div.foot a,
[data-theme="dark"] div.foot a,
.gradio-container.dark div.foot a {
  color: #7ec8ff !important;
}
"""

try:
    APP_THEME = gr.themes.Glass(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.blue,
        font=[gr.themes.GoogleFont("Source Sans 3"), "ui-sans-serif", "sans-serif"],
    ).set(
        body_background_fill="#e8eef7",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_label_text_weight="600",
        input_border_width="1px",
    )
except Exception:
    APP_THEME = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        font=[gr.themes.GoogleFont("Source Sans 3"), "sans-serif"],
        radius_size=gr.themes.sizes.radius_md,
        spacing_size=gr.themes.sizes.spacing_md,
    )

_GRADIO_MAJOR = int(gr.__version__.split(".", maxsplit=1)[0])
_BLOCKS_KW: dict = {"title": f"{SOFTWARE_NAME} · Kumar Group (UNSW)"}
_LAUNCH_THEME_KW: dict = {}
if _GRADIO_MAJOR >= 6:
    _LAUNCH_THEME_KW = {"theme": APP_THEME, "css": CUSTOM_CSS}
else:
    _BLOCKS_KW["theme"] = APP_THEME
    _BLOCKS_KW["css"] = CUSTOM_CSS

_HERO_HTML = f"""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-main">
      <p class="hero-kicker">Kumar Research Group · Computational drug discovery · UNSW Sydney</p>
      <h1>{SOFTWARE_NAME}</h1>
      <p class="sub">
        An integrated command-line and web platform for <strong>physicochemical profiling</strong> and
        <strong>deep learning-based half-life prediction</strong> of therapeutic peptides. The workbench supports
        single-sequence and batch prediction, optional profiling, and publication-quality figures.
        Use the <strong>How to cite</strong> tab for BibTeX, CITATION.cff, and a downloadable bundle.
      </p>
      <div class="hero-actions">
        <a class="hero-link" href="{GITHUB_REPO}" target="_blank" rel="noopener noreferrer">GitHub repository</a>
      </div>
    </div>
    <aside class="hero-aside" aria-label="Model summary">
      <div class="hero-card">
        <span class="hero-card-title">Models</span>
        <p class="hero-card-model-hint" style="color: #ffffff !important;">Choose a preset under <strong style="color: #ffffff !important;">Half-life prediction model</strong>: CNN–BiLSTM + physchem (emb., default) or CNN + physchem (emb.).</p>
      </div>
    </aside>
  </div>
</div>
"""

BIBTEX_FOR_UI = _read_text(CITATION_BIB)

with gr.Blocks(**_BLOCKS_KW) as demo:
    gr.HTML(_HERO_HTML)

    with gr.Column(elem_classes=["app-shell"]):
        model_dd = gr.Dropdown(
            label="Half-life prediction model",
            choices=PRESET_DROPDOWN_LABELS,
            value=PRESET_DROPDOWN_LABELS[0],
        )

        with gr.Tabs():
            with gr.Tab("Single sequence"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        seq_in = gr.Textbox(
                            label="Sequence",
                            placeholder="e.g. KWKLFKKIGAVLKVL",
                            lines=3,
                            value="KWKLFKKIGAVLKVL",
                        )
                        phy_chk = gr.Checkbox(
                            label="Physicochemical profile (recommended)",
                            value=True,
                        )
                        fig_chk = gr.Checkbox(
                            label="Publication figure suite (high-resolution PNG + ZIP)",
                            value=True,
                        )
                        run_btn = gr.Button("Run prediction", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.HTML('<p class="section-heading">Results</p>')
                        out_md = gr.Markdown()
                        out_tbl = gr.Dataframe(label="Property table", wrap=True, max_height=280)
                        gr.HTML('<p class="section-heading">Amino acid composition</p>')
                        aa_comp_img = gr.Image(
                            label="Percentage by residue (±1 SE) — standard AAs with count > 0",
                            type="filepath",
                            elem_classes=["aa-composition-chart"],
                        )
                        gr.HTML('<p class="section-heading">Publication figures</p>')
                        fig_status = gr.Markdown()
                        single_gallery = gr.Gallery(
                            label="Panels (click for fullscreen preview)",
                            columns=1,
                            rows=1,
                            height=880,
                            allow_preview=True,
                            object_fit="contain",
                            elem_classes=["figure-gallery-wrap", "pro-gallery"],
                        )
                        single_fig_zip = gr.File(
                            label="Download figure bundle (ZIP, full-resolution PNGs)",
                            interactive=False,
                            visible=True,
                        )

                run_btn.click(
                    fn=predict_single,
                    inputs=[model_dd, seq_in, phy_chk, fig_chk],
                    outputs=[
                        out_md,
                        out_tbl,
                        aa_comp_img,
                        fig_status,
                        single_gallery,
                        single_fig_zip,
                    ],
                )

            with gr.Tab("Batch (paste)"):
                gr.Markdown(
                    "One sequence per line · empty lines and `#` comments ignored · **max 200** · "
                    "try **≥6 peptides** with physicochemical profiling for the richest figure set."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        batch_in = gr.Textbox(
                            label="Sequences",
                            lines=14,
                            placeholder="KWKLFKKIGAVLKVL\nGIGKFLHSAKKFGKAFVGEIMNS",
                        )
                        phy_chk_b = gr.Checkbox(label="Physicochemical profile (recommended)", value=True)
                        fig_chk_b = gr.Checkbox(
                            label="Publication figure suite (high-resolution PNG + ZIP)",
                            value=True,
                        )
                        batch_btn = gr.Button("Run batch", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.HTML('<p class="section-heading">Summary & table</p>')
                        batch_summary = gr.Markdown()
                        batch_df = gr.Dataframe(label="Batch results", wrap=True, max_height=320)
                        batch_file = gr.File(label="Results CSV", visible=True)
                        gr.HTML('<p class="section-heading">Amino acid composition</p>')
                        batch_aa_comp_img = gr.Image(
                            label="Pooled % by residue (±1 SE) — standard AAs with count > 0",
                            type="filepath",
                            elem_classes=["aa-composition-chart"],
                        )
                        gr.HTML('<p class="section-heading">Publication figures</p>')
                        batch_fig_status = gr.Markdown()
                        batch_gallery = gr.Gallery(
                            label="Panels (click for fullscreen preview)",
                            columns=1,
                            rows=1,
                            height=880,
                            allow_preview=True,
                            object_fit="contain",
                            elem_classes=["figure-gallery-wrap", "pro-gallery"],
                        )
                        batch_fig_zip = gr.File(
                            label="Download figure bundle (ZIP, full-resolution PNGs)",
                            interactive=False,
                            visible=True,
                        )

                def _batch_wrapper(mdl, text, inc, gen_fig):
                    df, sm, path, aa_png, fig_md, gallery, zpath = predict_batch(
                        mdl, text, inc, gen_fig
                    )
                    return (
                        sm,
                        df,
                        gr.update(value=path, visible=bool(path)),
                        gr.update(value=aa_png, visible=bool(aa_png)),
                        fig_md,
                        gallery,
                        gr.update(value=zpath, visible=bool(zpath)),
                    )

                batch_btn.click(
                    fn=_batch_wrapper,
                    inputs=[model_dd, batch_in, phy_chk_b, fig_chk_b],
                    outputs=[
                        batch_summary,
                        batch_df,
                        batch_file,
                        batch_aa_comp_img,
                        batch_fig_status,
                        batch_gallery,
                        batch_fig_zip,
                    ],
                )

            with gr.Tab("How to cite"):
                gr.Markdown(
                    f'<div class="cite-panel"><strong>Plain text</strong><br/><br/>'
                    f'<code class="cite-panel-code">{html.escape(CITATION_LINE)}</code></div>'
                )
                gr.Markdown("**BibTeX** — copy from the code box or download `.bib`:")
                gr.Code(
                    value=BIBTEX_FOR_UI,
                    language=None,
                    label="BibTeX",
                    lines=16,
                    interactive=False,
                )
                gr.Markdown("**Downloads** (same files as on GitHub):")
                with gr.Row():
                    bib_file = gr.File(
                        label="CITATION.bib",
                        value=str(CITATION_BIB) if CITATION_BIB.is_file() else None,
                        interactive=False,
                    )
                    cff_file = gr.File(
                        label="CITATION.cff (GitHub)",
                        value=str(CITATION_CFF) if CITATION_CFF.is_file() else None,
                        interactive=False,
                    )
                zip_btn = gr.Button("Build citation ZIP (BibTeX + CFF + CITATION.txt)", variant="secondary")
                zip_out = gr.File(label="Download citation bundle", interactive=False)

                zip_btn.click(fn=_build_citation_zip, inputs=[], outputs=zip_out)

    gr.HTML(
        f"""
<div class="foot-cite">
  <p class="foot-cite-intro">{html.escape(CITATION_INTRO)}</p>
  <p class="foot-cite-body">{html.escape(CITATION_LINE)}</p>
</div>
"""
    )


def _on_hf_space() -> bool:
    """Hugging Face Spaces injects SPACE_* env vars; use safer public defaults there."""
    return any(
        os.environ.get(k)
        for k in (
            "SPACE_ID",
            "SPACE_AUTHOR_NAME",
            "SPACE_REPO_NAME",
            "SPACE_HARDWARE",
            "SPACE_SYSTEM_VERSION",
        )
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue(default_concurrency_limit=2)
    _raw_show = os.environ.get("GRADIO_SHOW_ERROR")
    if _raw_show is not None and str(_raw_show).strip() != "":
        _show_err = str(_raw_show).strip().lower() in ("1", "true", "yes")
    else:
        _show_err = not _on_hf_space()

    _launch_kw: Dict[str, Any] = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "show_error": _show_err,
        **_LAUNCH_THEME_KW,
    }
    if _on_hf_space() or os.environ.get("GRADIO_DISABLE_SSR", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        if "ssr_mode" in inspect.signature(demo.launch).parameters:
            _launch_kw["ssr_mode"] = False
    demo.launch(**_launch_kw)
