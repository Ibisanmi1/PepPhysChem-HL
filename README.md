---
title: AMP PhysioChemical Predictor
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: AMP half-life and physchem profiles for peptides (research).
---
# AMP_PhysioChem

A comprehensive computational platform that integrates established biochemical analysis methods with advanced deep-learning models for therapeutics peptide characterization

It is intended for research and education (e.g. open demos and reproducible workflows). Predictions are computational estimates, not clinical or regulatory advice.

**Hugging Face Space:** [Ibisanmi1/AMP_PhysioChemical_Predictor](https://huggingface.co/spaces/Ibisanmi1/AMP_PhysioChemical_Predictor). 

### Step 1: Clone or Navigate to Project Directory

git clone https://github.com/Ibisanmi1/AMP_PhysioChem_Predictor.git


```bash
cd /path/to/AMP_PhysioChem_Predictor
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

```bash
python run_AMP_PhysioChem_Predictor.py \
    --sequence "KWKLFKKIGAVLKVL" \
    --output "single_result.csv"
```


#### Example 2: Batch Analysis

Analyze multiple sequences from a CSV file:

```bash
python run_AMP_PhysioChem_Predictor.py \
    --input "example_peptides.csv" \
    --output "batch_results.csv"
```

**Note**: The default column name for sequences is `sequence`. If your CSV uses a different column name, specify it with `--sequence_col`:

```bash
python run_AMP_PhysioChem_Predictor.py \
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

#### Example 3: if you like to use other Checkpoint

```bash
python run_AMP_PhysioChem_Predictor.py \
    --model_path "checkpoints/Half_Life_cnn_bilstm_embedding_2.pt" \
    --sequence "KWKLFKKIGAVLKVL" \
    --output "single_result_model2.csv"
```


#### Example 5: Force CPU Usage

```bash
python run_AMP_PhysioChem_Predictor.py \
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
AMP_PhysioChem_Predictor/
├── run_AMP_PhysioChem_Predictor.py    # Main entry point
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

Ibisanmi TA, Bahatheg G,Mishra (Baishnab) SK,  Willcox M, Kumar N (2026). AMP_PhysioChem_Predictor: Comprehensive computational software for the prediction of physicochemical properties and antimicrobial peptide stability.....
