import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, linregress
import warnings
warnings.filterwarnings('ignore')

CHART_PRIMARY = "#2e6da4"
CHART_EDGE = "#1a3d5c"
CHART_FONT_SANS = [
    "Source Sans 3",
    "Segoe UI",
    "DejaVu Sans",
    "Arial",
    "Liberation Sans",
    "sans-serif",
]

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": CHART_FONT_SANS,
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})
sns.set_style("whitegrid")
sns.set_palette("deep")



project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

SOFTWARE_NAME = "PepPhysChem-HL"
SOFTWARE_FULL_TITLE = (
    "PepPhysChem-HL: An Integrated Command-Line and Web Platform for "
    "Physicochemical Profiling and Deep Learning-Based Half-Life Prediction "
    "of Therapeutic Peptides"
)

_ai_root_raw = (
    os.environ.get("PEPPHYSCHEM_HL_AI_ROOT")
    or os.environ.get("AMP_PHYSIOCHEM_AI_ROOT")
    or ""
).strip()
_ai_root_candidate = Path(_ai_root_raw).expanduser() if _ai_root_raw else None
if _ai_root_candidate is not None and _ai_root_candidate.is_dir():
    PEPPHYSOCHEM_HL_AI_ROOT = _ai_root_candidate
else:
    PEPPHYSOCHEM_HL_AI_ROOT = project_root.parent.parent / "AMP_PhysioChem_AI"

# Backward-compatible alias for existing deployments and documentation.
AMP_PHYSIOCHEM_AI_ROOT = PEPPHYSOCHEM_HL_AI_ROOT


def _default_results_output_dir() -> Path:
    """Directory for relative --output paths: always <this package>/output/."""
    out = project_root / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


DEFAULT_HYBRID_CHECKPOINT_NAME = "Half_Life_cnn_bilstm_embedding_physchem_run1.pt"
HYBRID_CHECKPOINT_BASENAMES = (DEFAULT_HYBRID_CHECKPOINT_NAME,)
DEFAULT_HYBRID_TRAINING_CONFIG = (
    project_root / "training_logs" / "1_cnn_bilstm_hybrid_physchem_matrix" / "training_config.json"
)


def default_hybrid_checkpoint_path(root: Optional[Path] = None) -> Path:
    """Absolute path to the recommended hybrid weights under <root>/checkpoints/."""
    base = root if root is not None else project_root
    return base / "checkpoints" / DEFAULT_HYBRID_CHECKPOINT_NAME


def _resolve_default_hybrid_checkpoint() -> Path:
    """
    Best hybrid weights: local checkpoints first, then PEPPHYSOCHEM_HL_AI_ROOT/checkpoints
    so a slim PepPhysChem-HL clone can use the sibling training repo.
    """
    search_roots = [project_root]
    if PEPPHYSOCHEM_HL_AI_ROOT.is_dir() and PEPPHYSOCHEM_HL_AI_ROOT.resolve() != project_root.resolve():
        search_roots.append(PEPPHYSOCHEM_HL_AI_ROOT)
    for root in search_roots:
        ck = root / "checkpoints"
        for name in HYBRID_CHECKPOINT_BASENAMES:
            p = ck / name
            if p.is_file():
                return p
    return project_root / "checkpoints" / HYBRID_CHECKPOINT_BASENAMES[0]


from src.embedding_models import get_embedding_model
from src.embedding_datasets import (
    EmbeddingPeptideDataset,
    embedding_collate_fn,
    EmbeddingPhysioChemDataset,
)
from src.physicochemical_analyzer import PhysicochemicalAnalyzer


class ComprehensiveAnalysis:
    """
    Publication-quality comprehensive analysis of peptide analysis results.
    Generates high-quality figures and statistical summaries suitable for manuscripts.
    """

    def __init__(self, output_dir: Path):
        """Initialize analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


        self.key_properties = {
            'net_charge_pH7': 'Net Charge (pH 7.0)',
            'half_life': 'Half-Life (minutes)',
            'hydrophobicity_kd_mean': 'Hydrophobicity (KD mean)',
            'percentage_hydrophobicity': 'Hydrophobic Percentage (%)',
            'hydrophobicity_kd_sum': 'Hydrophobicity (KD sum)',
            'amphipathicity_index': 'Amphipathicity Index',
            'amphipathic_patterns': 'Amphipathic Patterns',
            'helix_fraction': 'Helix Fraction',
            'length': 'Sequence Length',
            'isoelectric_point': 'Isoelectric Point (pI)'
        }

    def analyze_results(self, df: pd.DataFrame, output_prefix: str = "analysis"):
        """
        Perform comprehensive publication-quality analysis.

        Args:
            df: Results DataFrame
            output_prefix: Prefix for output files
        """
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE STATISTICAL ANALYSIS (Publication Quality)")
        print("="*80)

        if 'half_life' not in df.columns:
            print("⚠️  Warning: 'half_life' column not found. Analysis may be limited.")
            return


        self._generate_statistical_summary(df, output_prefix)


        self._plot_distributions(df, output_prefix)


        self._analyze_correlations(df, output_prefix)


        self._plot_half_life_relationships(df, output_prefix)


        self._analyze_hydrophobic_properties(df, output_prefix)


        self._analyze_charge_amphipathicity(df, output_prefix)


        self._analyze_structural_properties(df, output_prefix)


        self._generate_summary_report(df, output_prefix)

        print(f"\n✅ Comprehensive analysis complete!")
        print(f"   Figures saved to: {self.output_dir}")
        print(f"   Report saved to: {self.output_dir / f'{output_prefix}_analysis_report.txt'}")

    def _generate_statistical_summary(self, df: pd.DataFrame, prefix: str):
        """Generate comprehensive statistical summary."""
        print("\n📈 Statistical Summary:")

        stats_data = []
        for prop, label in self.key_properties.items():
            if prop in df.columns:
                col_data = df[prop].dropna()
                if len(col_data) > 0:
                    stats_data.append({
                        'Property': label,
                        'Mean': col_data.mean(),
                        'Median': col_data.median(),
                        'Std': col_data.std(),
                        'Min': col_data.min(),
                        'Max': col_data.max(),
                        'Q1': col_data.quantile(0.25),
                        'Q3': col_data.quantile(0.75)
                    })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            print(stats_df.to_string(index=False))
            stats_df.to_csv(self.output_dir / f'{prefix}_statistics.csv', index=False)

    def _plot_distributions(self, df: pd.DataFrame, prefix: str):
        """Create publication-quality distribution plots."""
        print("\n📊 Generating distribution plots...")


        available_props = {k: v for k, v in self.key_properties.items() if k in df.columns}

        if not available_props:
            return

        n_props = len(available_props)
        n_cols = 3
        n_rows = (n_props + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_props == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        fig.suptitle("Distribution of Key Peptide Properties", fontsize=13, fontweight="bold", y=0.995)

        for idx, (prop, label) in enumerate(available_props.items()):
            ax = axes[idx]
            data = df[prop].dropna()

            if len(data) > 0:
                ax.hist(
                    data,
                    bins=20,
                    density=True,
                    alpha=1.0,
                    color=CHART_PRIMARY,
                    edgecolor=CHART_EDGE,
                    linewidth=0.8,
                )

                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    ax.plot(
                        x_range,
                        kde(x_range),
                        color=CHART_EDGE,
                        linewidth=2.5,
                        label="KDE",
                    )
                except Exception:
                    pass

                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(
                    mean_val,
                    color=CHART_EDGE,
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.2f}",
                )
                ax.axvline(
                    median_val,
                    color=CHART_PRIMARY,
                    linestyle=":",
                    linewidth=2,
                    label=f"Median: {median_val:.2f}",
                )

                ax.set_xlabel(label, fontsize=12, fontweight="bold")
                ax.set_ylabel("Density", fontsize=12, fontweight="bold")
                ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
                ax.legend(fontsize=10, framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--')


        for idx in range(len(available_props), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(self.output_dir / f'{prefix}_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {prefix}_distributions.png")

    def _analyze_correlations(self, df: pd.DataFrame, prefix: str):
        """Analyze correlations with half-life."""
        print("\n🔗 Correlation Analysis with Half-Life...")

        if 'half_life' not in df.columns:
            return


        correlations = []
        for prop, label in self.key_properties.items():
            if prop in df.columns and prop != 'half_life':
                data = df[[prop, 'half_life']].dropna()
                if len(data) > 3:
                    try:
                        pearson_r, pearson_p = pearsonr(data[prop], data['half_life'])
                        spearman_r, spearman_p = spearmanr(data[prop], data['half_life'])
                        correlations.append({
                            'Property': label,
                            'Pearson r': pearson_r,
                            'Pearson p': pearson_p,
                            'Spearman ρ': spearman_r,
                            'Spearman p': spearman_p
                        })
                    except:
                        continue

        if correlations:
            corr_df = pd.DataFrame(correlations)
            corr_df = corr_df.sort_values('Pearson r', key=abs, ascending=False)

            print("\n   Top Correlations with Half-Life:")
            print(corr_df.to_string(index=False))
            corr_df.to_csv(self.output_dir / f'{prefix}_correlations.csv', index=False)


            fig, ax = plt.subplots(figsize=(12, 8))

            y_pos = np.arange(len(corr_df))
            colors = ['red' if r < 0 else 'blue' for r in corr_df['Pearson r']]

            bars = ax.barh(
                y_pos,
                corr_df["Pearson r"],
                color=colors,
                alpha=0.7,
                edgecolor=CHART_EDGE,
                linewidth=0.8,
            )


            for i, (idx, row) in enumerate(corr_df.iterrows()):
                sig = '***' if row['Pearson p'] < 0.001 else '**' if row['Pearson p'] < 0.01 else '*' if row['Pearson p'] < 0.05 else 'ns'
                ax.text(row['Pearson r'], i, f" {sig}", va='center', fontsize=10, fontweight='bold')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(corr_df['Property'], fontsize=11)
            ax.set_xlabel(
                "Pearson Correlation Coefficient (r)",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_title(
                "Correlation of Key Properties with Half-Life",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            ax.set_xlim(-1, 1)

            plt.tight_layout()
            plt.savefig(self.output_dir / f'{prefix}_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: {prefix}_correlations.png")

    def _plot_half_life_relationships(self, df: pd.DataFrame, prefix: str):
        """Plot key relationships with half-life."""
        print("\n📈 Generating Half-Life Relationship Plots...")

        if 'half_life' not in df.columns:
            return


        plot_props = {
            'net_charge_pH7': 'Net Charge (pH 7.0)',
            'hydrophobicity_kd_mean': 'Hydrophobicity (KD mean)',
            'percentage_hydrophobicity': 'Hydrophobic Percentage (%)',
            'amphipathicity_index': 'Amphipathicity Index',
            'amphipathic_patterns': 'Amphipathic Patterns',
            'helix_fraction': 'Helix Fraction',
            'length': 'Sequence Length',
            'isoelectric_point': 'Isoelectric Point (pI)'
        }

        available_props = {k: v for k, v in plot_props.items() if k in df.columns}

        if not available_props:
            return

        n_props = len(available_props)
        n_cols = 3
        n_rows = (n_props + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_props == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        fig.suptitle(
            "Key Properties vs Half-Life Relationships",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )

        for idx, (prop, label) in enumerate(available_props.items()):
            ax = axes[idx]
            data = df[[prop, 'half_life']].dropna()

            if len(data) > 2:

                ax.scatter(data[prop], data['half_life'], alpha=0.6, s=80,
                          edgecolors=CHART_EDGE, linewidth=0.8, color=CHART_PRIMARY, zorder=3)


                try:
                    slope, intercept, r_value, p_value, std_err = linregress(data[prop], data['half_life'])
                    x_line = np.linspace(data[prop].min(), data[prop].max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'Linear fit (r={r_value:.3f})', zorder=2)


                    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                    stats_text = f'r = {r_value:.3f} {sig}\np = {p_value:.3e}'
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=11, verticalalignment='top', fontweight='bold')
                except:
                    pass

                ax.set_xlabel(label, fontsize=12, fontweight="bold")
                ax.set_ylabel("Half-Life (minutes)", fontsize=12, fontweight="bold")
                ax.set_title(f"{label} vs Half-Life", fontsize=13, fontweight="bold", pad=10)
                ax.legend(fontsize=10, framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--', zorder=1)


        for idx in range(len(available_props), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(self.output_dir / f'{prefix}_half_life_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {prefix}_half_life_relationships.png")

    def _analyze_hydrophobic_properties(self, df: pd.DataFrame, prefix: str):
        """Analyze hydrophobic properties."""
        print("\n💧 Analyzing Hydrophobic Properties...")

        hydrophobic_props = ['hydrophobicity_kd_mean', 'percentage_hydrophobicity',
                            'hydrophobicity_kd_sum', 'hydrophobic_count']
        available_props = [p for p in hydrophobic_props if p in df.columns]

        if not available_props or 'half_life' not in df.columns:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Hydrophobic Properties Analysis",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        axes = axes.flatten()

        for idx, prop in enumerate(available_props[:4]):
            ax = axes[idx]
            data = df[[prop, 'half_life']].dropna()

            if len(data) > 2:
                ax.scatter(data[prop], data['half_life'], alpha=0.6, s=80,
                          edgecolors=CHART_EDGE, linewidth=0.8, color=CHART_PRIMARY)


                try:
                    slope, intercept, r_value, p_value, _ = linregress(data[prop], data['half_life'])
                    x_line = np.linspace(data[prop].min(), data[prop].max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'r={r_value:.3f}')
                except:
                    pass

                label = self.key_properties.get(prop, prop.replace('_', ' ').title())
                ax.set_xlabel(label, fontsize=12, fontweight="bold")
                ax.set_ylabel("Half-Life (minutes)", fontsize=12, fontweight="bold")
                ax.set_title(f"{label} vs Half-Life", fontsize=13, fontweight="bold")
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')


        for idx in range(len(available_props), 4):
            axes[idx].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(self.output_dir / f'{prefix}_hydrophobic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {prefix}_hydrophobic_analysis.png")

    def _analyze_charge_amphipathicity(self, df: pd.DataFrame, prefix: str):
        """Analyze charge and amphipathicity properties."""
        print("\n⚡ Analyzing Charge and Amphipathicity...")

        if 'half_life' not in df.columns:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Charge and Amphipathicity Analysis",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )


        if 'net_charge_pH7' in df.columns:
            ax = axes[0, 0]
            data = df[['net_charge_pH7', 'half_life']].dropna()
            if len(data) > 2:
                ax.scatter(data['net_charge_pH7'], data['half_life'], alpha=0.6, s=80,
                          edgecolors=CHART_EDGE, linewidth=0.8, color=CHART_PRIMARY)
                try:
                    slope, intercept, r_value, p_value, _ = linregress(data['net_charge_pH7'], data['half_life'])
                    x_line = np.linspace(data['net_charge_pH7'].min(), data['net_charge_pH7'].max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'r={r_value:.3f}')
                except:
                    pass
                ax.set_xlabel("Net Charge (pH 7.0)", fontsize=12, fontweight="bold")
                ax.set_ylabel("Half-Life (minutes)", fontsize=12, fontweight="bold")
                ax.set_title("Net Charge vs Half-Life", fontsize=13, fontweight="bold")
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')


        if 'amphipathicity_index' in df.columns:
            ax = axes[0, 1]
            data = df[['amphipathicity_index', 'half_life']].dropna()
            if len(data) > 2:
                ax.scatter(data['amphipathicity_index'], data['half_life'], alpha=0.6, s=80,
                          edgecolors=CHART_EDGE, linewidth=0.8, color=CHART_PRIMARY)
                try:
                    slope, intercept, r_value, p_value, _ = linregress(data['amphipathicity_index'], data['half_life'])
                    x_line = np.linspace(data['amphipathicity_index'].min(), data['amphipathicity_index'].max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'r={r_value:.3f}')
                except:
                    pass
                ax.set_xlabel("Amphipathicity Index", fontsize=12, fontweight="bold")
                ax.set_ylabel("Half-Life (minutes)", fontsize=12, fontweight="bold")
                ax.set_title(
                    "Amphipathicity Index vs Half-Life",
                    fontsize=13,
                    fontweight="bold",
                )
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')


        if 'amphipathic_patterns' in df.columns:
            ax = axes[1, 0]
            data = df[['amphipathic_patterns', 'half_life']].dropna()
            if len(data) > 2:
                ax.scatter(data['amphipathic_patterns'], data['half_life'], alpha=0.6, s=80,
                          edgecolors=CHART_EDGE, linewidth=0.8, color=CHART_PRIMARY)
                try:
                    slope, intercept, r_value, p_value, _ = linregress(data['amphipathic_patterns'], data['half_life'])
                    x_line = np.linspace(data['amphipathic_patterns'].min(), data['amphipathic_patterns'].max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'r={r_value:.3f}')
                except:
                    pass
                ax.set_xlabel("Amphipathic Patterns", fontsize=12, fontweight="bold")
                ax.set_ylabel("Half-Life (minutes)", fontsize=12, fontweight="bold")
                ax.set_title(
                    "Amphipathic Patterns vs Half-Life",
                    fontsize=13,
                    fontweight="bold",
                )
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')


        if 'net_charge_pH7' in df.columns and 'amphipathicity_index' in df.columns:
            ax = axes[1, 1]
            data = df[['net_charge_pH7', 'amphipathicity_index', 'half_life']].dropna()
            if len(data) > 2:
                scatter = ax.scatter(data['net_charge_pH7'], data['amphipathicity_index'],
                                   c=data['half_life'], s=100, alpha=0.7,
                                   edgecolors=CHART_EDGE, linewidth=0.8, cmap='viridis')
                ax.set_xlabel("Net Charge (pH 7.0)", fontsize=12, fontweight="bold")
                ax.set_ylabel("Amphipathicity Index", fontsize=12, fontweight="bold")
                ax.set_title(
                    "Charge vs Amphipathicity (colored by Half-Life)",
                    fontsize=13,
                    fontweight="bold",
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Half-Life (minutes)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(self.output_dir / f'{prefix}_charge_amphipathicity.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {prefix}_charge_amphipathicity.png")

    def _analyze_structural_properties(self, df: pd.DataFrame, prefix: str):
        """Analyze structural properties."""
        print("\n🧬 Analyzing Structural Properties...")

        if 'half_life' not in df.columns:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Structural Properties Analysis",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )


        if 'helix_fraction' in df.columns:
            ax = axes[0, 0]
            data = df[['helix_fraction', 'half_life']].dropna()
            if len(data) > 2:
                ax.scatter(data['helix_fraction'], data['half_life'], alpha=0.6, s=80,
                          edgecolors=CHART_EDGE, linewidth=0.8, color=CHART_PRIMARY)
                try:
                    slope, intercept, r_value, p_value, _ = linregress(data['helix_fraction'], data['half_life'])
                    x_line = np.linspace(data['helix_fraction'].min(), data['helix_fraction'].max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'r={r_value:.3f}')
                except:
                    pass
                ax.set_xlabel("Helix Fraction", fontsize=12, fontweight="bold")
                ax.set_ylabel("Half-Life (minutes)", fontsize=12, fontweight="bold")
                ax.set_title("Helix Fraction vs Half-Life", fontsize=13, fontweight="bold")
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')


        if 'length' in df.columns:
            ax = axes[0, 1]
            data = df[['length', 'half_life']].dropna()
            if len(data) > 2:
                ax.scatter(data['length'], data['half_life'], alpha=0.6, s=80,
                          edgecolors=CHART_EDGE, linewidth=0.8, color=CHART_PRIMARY)
                try:
                    slope, intercept, r_value, p_value, _ = linregress(data['length'], data['half_life'])
                    x_line = np.linspace(data['length'].min(), data['length'].max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'r={r_value:.3f}')
                except:
                    pass
                ax.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
                ax.set_ylabel("Half-Life (minutes)", fontsize=12, fontweight="bold")
                ax.set_title("Sequence Length vs Half-Life", fontsize=13, fontweight="bold")
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')


        if 'isoelectric_point' in df.columns:
            ax = axes[1, 0]
            data = df[['isoelectric_point', 'half_life']].dropna()
            if len(data) > 2:
                ax.scatter(data['isoelectric_point'], data['half_life'], alpha=0.6, s=80,
                          edgecolors=CHART_EDGE, linewidth=0.8, color=CHART_PRIMARY)
                try:
                    slope, intercept, r_value, p_value, _ = linregress(data['isoelectric_point'], data['half_life'])
                    x_line = np.linspace(data['isoelectric_point'].min(), data['isoelectric_point'].max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'r={r_value:.3f}')
                except:
                    pass
                ax.set_xlabel("Isoelectric Point (pI)", fontsize=12, fontweight="bold")
                ax.set_ylabel("Half-Life (minutes)", fontsize=12, fontweight="bold")
                ax.set_title(
                    "Isoelectric Point vs Half-Life",
                    fontsize=13,
                    fontweight="bold",
                )
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')


        if 'helix_fraction' in df.columns and 'length' in df.columns:
            ax = axes[1, 1]
            data = df[['helix_fraction', 'length', 'half_life']].dropna()
            if len(data) > 2:
                scatter = ax.scatter(data['helix_fraction'], data['length'],
                                   c=data['half_life'], s=100, alpha=0.7,
                                   edgecolors=CHART_EDGE, linewidth=0.8, cmap='viridis')
                ax.set_xlabel("Helix Fraction", fontsize=12, fontweight="bold")
                ax.set_ylabel("Sequence Length", fontsize=12, fontweight="bold")
                ax.set_title(
                    "Helix Fraction vs Length (colored by Half-Life)",
                    fontsize=13,
                    fontweight="bold",
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Half-Life (minutes)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(self.output_dir / f'{prefix}_structural_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {prefix}_structural_analysis.png")

    def _generate_summary_report(self, df: pd.DataFrame, prefix: str):
        """Generate comprehensive text report."""
        print("\n📝 Generating Summary Report...")

        report_path = self.output_dir / f'{prefix}_analysis_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE PEPTIDE ANALYSIS REPORT\n")
            f.write("Publication-Quality Statistical Analysis\n")
            f.write("="*80 + "\n\n")

            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Peptides Analyzed: {len(df)}\n\n")


            if 'half_life' in df.columns:
                f.write("HALF-LIFE STATISTICS\n")
                f.write("-"*80 + "\n")
                hl = df['half_life'].dropna()
                f.write(f"Mean:        {hl.mean():.4f} minutes\n")
                f.write(f"Median:      {hl.median():.4f} minutes\n")
                f.write(f"Std Dev:     {hl.std():.4f} minutes\n")
                f.write(f"Min:         {hl.min():.4f} minutes\n")
                f.write(f"Max:         {hl.max():.4f} minutes\n")
                f.write(f"Q1 (25%):    {hl.quantile(0.25):.4f} minutes\n")
                f.write(f"Q3 (75%):    {hl.quantile(0.75):.4f} minutes\n")
                f.write(f"IQR:         {hl.quantile(0.75) - hl.quantile(0.25):.4f} minutes\n")
                f.write(f"Skewness:    {hl.skew():.4f}\n")
                f.write(f"Kurtosis:    {hl.kurtosis():.4f}\n\n")


            f.write("KEY PROPERTIES SUMMARY\n")
            f.write("-"*80 + "\n")
            for prop, label in self.key_properties.items():
                if prop in df.columns:
                    col_data = df[prop].dropna()
                    if len(col_data) > 0:
                        f.write(f"{label:30s}: Mean = {col_data.mean():10.4f}, Std = {col_data.std():10.4f}\n")
            f.write("\n")


            if 'half_life' in df.columns:
                f.write("CORRELATIONS WITH HALF-LIFE\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Property':<30} {'Pearson r':<15} {'P-value':<15} {'Significance':<15}\n")
                f.write("-"*80 + "\n")

                correlations = []
                for prop, label in self.key_properties.items():
                    if prop in df.columns and prop != 'half_life':
                        data = df[[prop, 'half_life']].dropna()
                        if len(data) > 3:
                            try:
                                r, p = pearsonr(data[prop], data['half_life'])
                                correlations.append((label, r, p))
                            except:
                                continue

                correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                for label, r, p in correlations[:15]:
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    f.write(f"{label:<30} {r:>14.4f} {p:>14.4e} {sig:<15}\n")
                f.write("\n")


            if 'half_life' in df.columns:
                top_5 = df.nlargest(5, 'half_life')
                bottom_5 = df.nsmallest(5, 'half_life')

                f.write("TOP 5 PEPTIDES (Highest Half-Life)\n")
                f.write("-"*80 + "\n")
                for idx, row in top_5.iterrows():
                    id_val = row.get('id', 'N/A')
                    f.write(f"ID: {id_val}, Half-Life: {row['half_life']:.2f} minutes\n")
                f.write("\n")

                f.write("BOTTOM 5 PEPTIDES (Lowest Half-Life)\n")
                f.write("-"*80 + "\n")
                for idx, row in bottom_5.iterrows():
                    id_val = row.get('id', 'N/A')
                    f.write(f"ID: {id_val}, Half-Life: {row['half_life']:.2f} minutes\n")
                f.write("\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"   ✓ Saved: {prefix}_analysis_report.txt")


class PepPhysChemHLPredictor:
    """
    PepPhysChem-HL analysis pipeline combining:
    - Half-life prediction using the default hybrid CNN–BiLSTM + physicochemical model
      (cnn_bilstm_physchem)
    - Comprehensive physicochemical property analysis for therapeutic peptides
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        training_config_path: Optional[str] = None,
    ):
        """
        Initialize the analysis pipeline.

        Args:
            model_path: Path to model checkpoint (default: hybrid CNN–BiLSTM physchem .pt)
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
            training_config_path: Optional JSON next to a custom checkpoint (when the .pt lives
                under checkpoints/ and config is under training_logs/.../training_config.json).
        """

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"🔧 Using device: {self.device}")


        if model_path is None:
            model_path = _resolve_default_hybrid_checkpoint()
            config_path = (
                DEFAULT_HYBRID_TRAINING_CONFIG
                if DEFAULT_HYBRID_TRAINING_CONFIG.is_file()
                else None
            )
        else:
            model_path = Path(model_path)
            if training_config_path:
                config_path = Path(training_config_path)
                if not config_path.is_file():
                    config_path = None
            else:
                config_path = model_path.parent / "training_config.json"
                if not config_path.exists():
                    config_path = None

        if not model_path.exists():
            raise FileNotFoundError(
                f"Hybrid CNN–BiLSTM checkpoint not found: {model_path}\n"
                f"  Expected `checkpoints/{DEFAULT_HYBRID_CHECKPOINT_NAME}` under "
                f"{project_root / 'checkpoints'} or {PEPPHYSOCHEM_HL_AI_ROOT / 'checkpoints'}.\n"
                f"  Override with --model_path or set PEPPHYSOCHEM_HL_AI_ROOT to your training-repo tree."
            )

        print(f"📦 Loading model from: {model_path}")
        self.model, self.config = self._load_model(model_path, config_path)


        print("🧪 Initializing physicochemical analyzer...")
        self.analyzer = PhysicochemicalAnalyzer()


        self.aa_to_idx = {aa: idx + 1 for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.max_length = self.config.get('max_length', 100)

        print("✅ Pipeline initialized successfully!")
        mt = self.config.get("model_type", "cnn_bilstm")
        ft = self.config.get("feature_type", "embedding")
        if mt == "cnn_bilstm_physchem":
            print(f"   Model: {mt} (hybrid CNN–BiLSTM + physicochemical; {ft})")
        else:
            print(f"   Model: {mt} ({ft})")
        print(f"   Max sequence length: {self.max_length}")

    def _load_model(self, model_path: Path, config_path: Optional[Path] = None):
        """Load trained model from checkpoint."""

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)


        if 'config' in checkpoint:
            config = checkpoint['config']
        elif config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:

            config = {
                'model_type': 'cnn_bilstm',
                'feature_type': 'embedding',
                'max_length': 100,
                'model_config': {
                    'vocab_size': 21,
                    'embedding_dim': 128,
                    'conv_channels': [64, 64, 64],
                    'kernel_sizes': [3, 3, 3],
                    'lstm_hidden_dim': 64,
                    'lstm_num_layers': 2,
                    'dropout_rate': 0.3,
                    'output_dim': 1
                }
            }


        model_config = config.get('model_config', {})
        mt = config.get('model_type', 'cnn_bilstm')
        if mt == 'cnn_bilstm_physchem':
            model = get_embedding_model(
                'cnn_bilstm_physchem',
                vocab_size=model_config.get('vocab_size', 21),
                embedding_dim=model_config.get('embedding_dim', 128),
                conv_channels=model_config.get('conv_channels', [64, 64]),
                kernel_sizes=model_config.get('kernel_sizes', [3, 3]),
                lstm_hidden_dim=model_config.get('lstm_hidden_dim', 64),
                lstm_num_layers=model_config.get('lstm_num_layers', 2),
                physchem_input_dim=model_config.get('physchem_input_dim', 53),
                physchem_hidden_dims=model_config.get('physchem_hidden_dims', [32, 64]),
                dropout_rate=model_config.get('dropout_rate', 0.3),
                output_dim=model_config.get('output_dim', 1),
            )
        else:
            model = get_embedding_model(
                mt,
                vocab_size=model_config.get('vocab_size', 21),
                embedding_dim=model_config.get('embedding_dim', 128),
                conv_channels=model_config.get('conv_channels', [64, 64, 64]),
                kernel_sizes=model_config.get('kernel_sizes', [3, 3, 3]),
                lstm_hidden_dim=model_config.get('lstm_hidden_dim', 64),
                lstm_num_layers=model_config.get('lstm_num_layers', 2),
                dropout_rate=model_config.get('dropout_rate', 0.3),
                output_dim=model_config.get('output_dim', 1),
            )


        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model, config

    def _tokenize_sequence(self, sequence: str) -> torch.Tensor:
        """Tokenize a single sequence for embedding model."""
        sequence = sequence.strip().upper()

        sequence = ''.join(aa for aa in sequence if aa in self.aa_to_idx)


        input_ids = [self.aa_to_idx.get(aa, 0) for aa in sequence]


        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            input_ids.extend([0] * (self.max_length - len(input_ids)))

        return torch.tensor([input_ids], dtype=torch.long).to(self.device)

    def _clean_sequence_aa(self, sequence: str) -> str:
        sequence = sequence.strip().upper()
        return ''.join(aa for aa in sequence if aa in self.aa_to_idx)

    def _physchem_tensor_batch(self, sequences: List[str]) -> torch.Tensor:
        """53-d physicochemical vectors aligned with training (fast path, no RDKit)."""
        rows = []
        for seq in sequences:
            seq = self._clean_sequence_aa(seq)
            t = EmbeddingPhysioChemDataset._extract_physchem_features(self.analyzer, seq)
            rows.append(t)
        return torch.stack(rows).to(self.device)

    def predict_half_life(self, sequence: str) -> float:
        """
        Predict half-life for a single sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            Predicted half-life value
        """

        if len(sequence) > self.max_length:
            print(f"⚠️  Warning: Sequence length ({len(sequence)}) exceeds max_length ({self.max_length}). Truncating.")


        input_ids = self._tokenize_sequence(sequence)


        with torch.no_grad():
            if self.config.get('model_type') == 'cnn_bilstm_physchem':
                seq_clean = self._clean_sequence_aa(sequence)
                physchem = EmbeddingPhysioChemDataset._extract_physchem_features(
                    self.analyzer, seq_clean
                )
                physchem = physchem.unsqueeze(0).to(self.device)
                prediction = self.model(input_ids, physchem)
            else:
                prediction = self.model(input_ids)
            half_life = prediction.squeeze().item()

        return half_life

    def predict_batch(self, sequences: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Predict half-life for multiple sequences.

        Args:
            sequences: List of amino acid sequences
            batch_size: Batch size for prediction

        Returns:
            Array of predicted half-life values
        """
        predictions = []


        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]


            batch_input_ids = []
            for seq in batch_sequences:
                seq = seq.strip().upper()

                seq = ''.join(aa for aa in seq if aa in self.aa_to_idx)
                input_ids = [self.aa_to_idx.get(aa, 0) for aa in seq]


                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                else:
                    input_ids.extend([0] * (self.max_length - len(input_ids)))

                batch_input_ids.append(input_ids)


            batch_tensor = torch.tensor(batch_input_ids, dtype=torch.long).to(self.device)


            with torch.no_grad():
                if self.config.get('model_type') == 'cnn_bilstm_physchem':
                    physchem = self._physchem_tensor_batch(batch_sequences)
                    batch_predictions = self.model(batch_tensor, physchem)
                else:
                    batch_predictions = self.model(batch_tensor)
                predictions.extend(batch_predictions.cpu().numpy().flatten())

        return np.array(predictions)

    def analyze_single(self, sequence: str, include_physchem: bool = True) -> Dict:
        """
        Comprehensive analysis of a single peptide.

        Args:
            sequence: Amino acid sequence
            include_physchem: Whether to include physicochemical analysis

        Returns:
            Dictionary with prediction and physicochemical properties
        """

        predicted_log = self.predict_half_life(sequence)

        predicted_minutes = 10 ** predicted_log

        result = {
            'sequence': sequence,
            'length': len(sequence),
            'half_life': predicted_minutes
        }

        if include_physchem:
            try:
                physchem_profile = self.analyzer.calculate_comprehensive_profile(sequence)
                result.update(physchem_profile)
            except Exception as e:
                print(f"⚠️  Warning: Physicochemical analysis failed: {e}")
                result['physchem_error'] = str(e)

        return result

    def analyze_batch(self, sequences: List[str], include_physchem: bool = True,
                     progress: bool = True) -> pd.DataFrame:
        """
        Comprehensive analysis of multiple peptides.

        Args:
            sequences: List of amino acid sequences
            include_physchem: Whether to include physicochemical analysis
            progress: Whether to show progress updates

        Returns:
            DataFrame with predictions and physicochemical properties
        """
        results = []

        if progress:
            print(f"📊 Processing {len(sequences)} sequences...")


        if progress:
            print("   Predicting half-lives...")
        half_lives_log = self.predict_batch(sequences)

        half_lives_minutes = 10 ** half_lives_log


        for i, (seq, half_life_min) in enumerate(zip(sequences, half_lives_minutes)):
            if progress and (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(sequences)} sequences...")

            result = {
                'sequence': seq,
                'length': len(seq),
                'half_life': half_life_min
            }

            if include_physchem:
                try:
                    physchem_profile = self.analyzer.calculate_comprehensive_profile(seq)
                    result.update(physchem_profile)
                except Exception as e:
                    if progress:
                        print(f"   ⚠️  Warning: Failed to analyze sequence {i+1}: {e}")
                    result['physchem_error'] = str(e)

            results.append(result)

        if progress:
            print(f"✅ Completed analysis of {len(sequences)} sequences")

        return pd.DataFrame(results)

    def analyze_from_file(self, input_file: str, output_file: str,
                         sequence_col: str = 'sequence', include_physchem: bool = True):
        """
        Analyze peptides from input file (CSV or FASTA).

        Args:
            input_file: Path to input file (CSV or FASTA)
            output_file: Path to output CSV file
            sequence_col: Column name for sequences (CSV only)
            include_physchem: Whether to include physicochemical analysis
        """

        if not Path(input_file).is_absolute():
            input_path = project_root / "input" / input_file
            if not input_path.exists():

                input_path = Path(input_file)
        else:
            input_path = Path(input_file)


        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}\n"
                                  f"  Searched in: {project_root / 'input'}\n"
                                  f"  Make sure the file exists in the input/ folder or use an absolute path.")

        if input_path.suffix.lower() == '.fasta' or input_path.suffix.lower() == '.fa':
            sequences, headers = self._read_fasta(str(input_path))
            print(f"📖 Read {len(sequences)} sequences from FASTA file: {input_path.name}")
        else:

            df = pd.read_csv(input_path)
            if sequence_col not in df.columns:
                raise ValueError(f"Column '{sequence_col}' not found in CSV. Available columns: {df.columns.tolist()}")
            sequences = df[sequence_col].astype(str).tolist()


            id_column = None
            for col_name in ['peptide_id', 'id', 'name', 'identifier', 'peptide_name', 'seq_id']:
                if col_name in df.columns:
                    id_column = col_name
                    break

            if id_column:
                headers = df[id_column].astype(str).tolist()
                print(f"📖 Read {len(sequences)} sequences from CSV file: {input_path.name}")
                print(f"   Using '{id_column}' column for IDs")
            else:
                headers = [f'seq_{i}' for i in range(len(sequences))]
                print(f"📖 Read {len(sequences)} sequences from CSV file: {input_path.name}")
                print(f"   No ID column found, using auto-generated IDs")


        sequences = [seq.strip().upper() for seq in sequences]


        results_df = self.analyze_batch(sequences, include_physchem=include_physchem)


        if headers is not None:
            results_df.insert(0, 'id', headers)


        column_order = [
            'id', 'sequence',
            'net_charge_pH7',
            'charged_count',
            'basic_count',
            'half_life',
            'charge_density',
            'hydrophobicity_kd_mean',
            'percentage_hydrophobicity',
            'hydrophobic_moment',
            'amphipathicity_index',
            'amphipathic_patterns',
            'helix_fraction',
            'length',
            'hydrophobic_count',
            'isoelectric_point',
            'sheet_fraction',
            'turn_fraction',
            'aromatic_percent',
            'aa_W_count',
            'aa_F_count',
            'aa_Y_count',
            'side_chain_volume_mean',
            'side_chain_volume_total',
            'instability_index',
            'gravy',
            'molecular_weight',
            'aromaticity',
            'hydrophobicity_kd_sum',
            'acidic_count',
            'hydrophilic_percent',
            'charged_percent',
            'aa_K_percent',
            'aa_R_percent',
            'aa_L_percent',
            'aa_I_percent',
            'aa_V_percent',
            'aa_G_percent',
            'aa_P_percent',
            'aa_C_count',
            'aa_C_percent',
            'logP',
            'tpsa',
            'num_h_donors',
            'num_h_acceptors',
            'fraction_csp3',
            'num_rotatable_bonds',
            'num_aromatic_rings'
        ]


        existing_columns = list(results_df.columns)
        ordered_columns = []


        for col in column_order:
            if col in existing_columns:
                ordered_columns.append(col)


        for col in existing_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)


        results_df = results_df[ordered_columns]


        if not Path(output_file).is_absolute():
            output_path = _default_results_output_dir() / output_file
        else:
            output_path = Path(output_file)


        output_path.parent.mkdir(parents=True, exist_ok=True)


        results_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")


        print("\n📈 Summary Statistics:")
        print(f"   Mean half-life: {results_df['half_life'].mean():.2f} minutes")
        print(f"   Std half-life: {results_df['half_life'].std():.2f} minutes")
        print(f"   Min half-life: {results_df['half_life'].min():.2f} minutes")
        print(f"   Max half-life: {results_df['half_life'].max():.2f} minutes")


        if include_physchem:
            print("\n" + "="*80)
            print("🔬 Starting Comprehensive Statistical Analysis (Publication Quality)...")
            print("="*80)
            analyzer = ComprehensiveAnalysis(output_path.parent)
            output_name = output_path.stem
            analyzer.analyze_results(results_df, output_prefix=output_name)

    def _read_fasta(self, fasta_file: str) -> tuple:
        """Read sequences from FASTA file."""
        sequences = []
        headers = []

        with open(fasta_file, 'r') as f:
            current_seq = []
            current_header = None

            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_header is not None:
                        sequences.append(''.join(current_seq))
                    current_header = line[1:].split()[0]
                    headers.append(current_header)
                    current_seq = []
                else:
                    current_seq.append(line)


            if current_header is not None:
                sequences.append(''.join(current_seq))

        return sequences, headers


def main():
    parser = argparse.ArgumentParser(
        description=f'{SOFTWARE_NAME} — peptide half-life and physicochemical analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_PepPhysChem_HL.py --sequence "KWKLFKKIGAVLKVL"
  python run_PepPhysChem_HL.py --input peptides.csv --output results.csv
  python run_PepPhysChem_HL.py --model_path checkpoints/custom_model.pt --sequence "ACDEFGHIK"
        """
    )

    parser.add_argument('--sequence', type=str, help='Single peptide sequence to analyze')
    parser.add_argument('--input', type=str, help='Input file (CSV or FASTA) for batch analysis (looks in input/ folder by default)')
    parser.add_argument(
        '--output',
        type=str,
        default='results.csv',
        help='Output CSV. Relative paths are written under ./output/ in this package (absolute paths unchanged).',
    )
    parser.add_argument('--sequence_col', type=str, default='sequence',
                       help='Column name for sequences in CSV (default: sequence)')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help=(
            'Path to .pt checkpoint (default: best hybrid CNN–BiLSTM+physchem run 1: '
            'Half_Life_cnn_bilstm_embedding_physchem_run1.pt under ./checkpoints '
            'or PEPPHYSOCHEM_HL_AI_ROOT/checkpoints)'
        ),
    )
    parser.add_argument(
        '--training_config',
        type=str,
        default=None,
        help=(
            'Optional path to training_config.json when --model_path points to a file in '
            'checkpoints/ (config not in the same folder). Used by the Gradio app for named presets.'
        ),
    )
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--no_physchem', action='store_true',
                       help='Skip physicochemical analysis (faster)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for predictions (default: 32)')

    args = parser.parse_args()


    if not args.sequence and not args.input:
        parser.error("Either --sequence or --input must be provided")

    if args.sequence and args.input:
        parser.error("Cannot specify both --sequence and --input")

    try:

        pipeline = PepPhysChemHLPredictor(
            model_path=args.model_path,
            device=args.device,
            training_config_path=args.training_config,
        )


        if args.sequence:

            print(f"\n🔬 Analyzing sequence: {args.sequence}")
            result = pipeline.analyze_single(args.sequence, include_physchem=not args.no_physchem)


            print("\n" + "="*80)
            print("RESULTS")
            print("="*80)
            print(f"Sequence: {result['sequence']}")
            print(f"Length: {result['length']}")
            print(f"Half-Life: {result['half_life']:.2f} minutes")

            if not args.no_physchem:
                print("\nPhysicochemical Properties:")
                print(f"  Molecular Weight: {result.get('molecular_weight', 'N/A'):.2f} Da")
                print(f"  Isoelectric Point: {result.get('isoelectric_point', 'N/A'):.4f}")
                print(f"  Net Charge (pH 7): {result.get('net_charge_pH7', 'N/A'):.4f}")
                print(f"  GRAVY: {result.get('gravy', 'N/A'):.4f}")
                print(f"  Hydrophobicity (KD mean): {result.get('hydrophobicity_kd_mean', 'N/A'):.4f}")
                print(f"  Instability Index: {result.get('instability_index', 'N/A'):.4f}")


            if not Path(args.output).is_absolute():
                output_path = _default_results_output_dir() / args.output
            else:
                output_path = Path(args.output)


            output_path.parent.mkdir(parents=True, exist_ok=True)


            df = pd.DataFrame([result])
            df.to_csv(output_path, index=False)
            print(f"\n💾 Results saved to: {output_path}")

        else:

            pipeline.analyze_from_file(
                args.input,
                args.output,
                sequence_col=args.sequence_col,
                include_physchem=not args.no_physchem
            )

        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Backward-compatible alias for scripts and deployments using the old class name.
AMPPhysioChemPredictor = PepPhysChemHLPredictor


if __name__ == "__main__":
    main()

