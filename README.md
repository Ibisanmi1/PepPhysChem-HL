

# PepPhysChem-HL

**PepPhysChem-HL: An Integrated Command-Line and Web Platform for Physicochemical Profiling and Deep Learning-Based Half-Life Prediction of Therapeutic Peptides**

A comprehensive computational platform that integrates established biochemical analysis methods with advanced deep-learning models for therapeutic peptide characterization.

It is intended for research and education (e.g. open demos and reproducible workflows). Predictions are computational estimates, not clinical or regulatory advice.

**Hugging Face Space:** [Ibisanmi1/PepPhysChem-HL](https://huggingface.co/spaces/Ibisanmi1/PepPhysChem-HL). 

### Step 1: Clone or Navigate to Project Directory

git clone https://github.com/Ibisanmi1/PepPhysChem-HL.git


```bash
cd /path/to/PepPhysChem-HL
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: RDKit installation may require additional steps:
- **macOS**: `conda install -c conda-forge rdkit`
- **Linux**: `conda install -c conda-forge rdkit` or use pip
- **Windows**: Use conda or follow RDKit installation guide



Usage

#### Example 1: Single Sequence Analysis

Uses the **default recommended checkpoint** automatically:
`checkpoints/Half_Life_cnn_bilstm_embedding_physchem_run1.pt` (hybrid CNN–BiLSTM + physicochemical matrix; top-ranked in `data/model_comparison.csv`).

**Training logs:** `training_logs/1_cnn_bilstm_hybrid_physchem_matrix/` (`training.log`, `training_config.json`, `final_results.json`, `epoch_metrics.csv`)

```bash
python run_PepPhysChem_HL.py \
    --sequence "KWKLFKKIGAVLKVL" \
    --output "single_result.csv"
```


#### Example 2: Batch Analysis

Analyze multiple sequences from a CSV file:

```bash
python run_PepPhysChem_HL.py \
    --input "example_peptides.csv" \
    --output "batch_results.csv"
```

**Note**: The default column name for sequences is `sequence`. If your CSV uses a different column name, specify it with `--sequence_col`:

```bash
python run_PepPhysChem_HL.py \
    --input "my_data.csv" \
    --sequence_col "peptide_sequence" \
    --output "batch_results.csv"
```

**Output**: CSV file + comprehensive statistical analysis including:
- Correlation plots (`*_correlations.png`)
- Distribution plots (`*_distributions.png`)
- Half-life relationship plots (`*_half_life_relationships.png`)
- Statistical summary (`*_statistics.csv`)
- Comprehensive report (`*_analysis_report.txt`)
- And more (see Output Files section below)

#### Example 3: Use an explicit checkpoint (optional)

Omit `--model_path` to use the default `Half_Life_cnn_bilstm_embedding_physchem_run1.pt` weights above.

```bash
python run_PepPhysChem_HL.py \
    --model_path "checkpoints/Half_Life_cnn_bilstm_embedding_physchem_run1.pt" \
    --training_config "training_logs/1_cnn_bilstm_hybrid_physchem_matrix/training_config.json" \
    --sequence "KWKLFKKIGAVLKVL" \
    --output "single_result_hybrid.csv"
```

**Alternative architecture** (embedding-only CNN–BiLSTM, no physicochemical block):

```bash
python run_PepPhysChem_HL.py \
    --model_path "checkpoints/Half_Life_cnn_bilstm_embedding_2.pt" \
    --sequence "KWKLFKKIGAVLKVL" \
    --output "single_result_embedding_only.csv"
```


#### Example 5: Force CPU Usage

```bash
python run_PepPhysChem_HL.py \
    --input "peptides.csv" \
    --device cpu \
    --output "results.csv"
```




## Input Formats

### CSV Format

The CSV file should contain at least one column with peptide sequences:

```csv
sequence
KWKLFKKIGAVLKVL
ACDEFGHIKLMNPQRSTVWY
```

**With additional columns** (optional):
```csv
id,sequence,notes
pep1,KWKLFKKIGAVLKVL,Test peptide 1
pep2,ACDEFGHIKLMNPQRSTVWY,Test peptide 2
```

The tool will automatically detect ID columns (`id`, `peptide_id`, `name`, `identifier`, `peptide_name`, `seq_id`).

### FASTA Format

Standard FASTA format is supported:

```
>peptide_1
KWKLFKKIGAVLKVL
>peptide_2
ACDEFGHIKLMNPQRSTVWY
```



## Directory Structure

```
PepPhysChem-HL/
├── run_PepPhysChem_HL.py          # Main entry point
├── run_analysis.sh             # Helper shell script
├── requirements.txt           # Python dependencies
├── checkpoints/                # Trained model files
├── input/                     # Input files (CSV/FASTA)
├── output/                     # Output files (results, plots, reports)
├── scripts/                    # Additional scripts
│   └── inference.py           # Direct inference script
├── src/                        # Source code modules
│   ├── models.py              # Neural network models
│   ├── data_processing.py     # Data loading and preprocessing
│   ├── physicochemical_analyzer.py  # Property calculations
│   └── ...
└── training/                   # Training scripts (see training/README.md)
```

If this pipeline contributes to your research, please cite:

Tope Abraham Ibisanmi, Ghayah Bahatheg, Shyam Kumar Mishra (Baishnab), Mark Willcox, and Naresh Kumar (2026). PepPhysChem-HL: An Integrated Command-Line and Web Platform for Physicochemical Profiling and Deep Learning-Based Half-Life Prediction of Therapeutic Peptides.
