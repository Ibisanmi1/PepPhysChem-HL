import os
import argparse
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.data_processing import PhysioChemDataProcessor, PhysioChemDataset, collate_fn
from src.physicochemical_analyzer import PhysicochemicalAnalyzer


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']


    if config['feature_type'] == 'physchem':
        model_kwargs = {'input_dim': config.get('input_dim', 50), 'output_dim': 1}
    elif config['feature_type'] == 'onehot':
        model_kwargs = {'input_channels': 20, 'output_dim': 1}
    elif config['feature_type'] == 'both':
        model_kwargs = {
            'seq_input_dim': 20,
            'physchem_input_dim': config.get('physchem_input_dim', 50),
            'output_dim': 1
        }


    model = get_model(config['model_type'], **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def predict_single(model, sequence: str, config: Dict, device: torch.device,
                  processor: PhysioChemDataProcessor) -> float:
    """Predict property for a single sequence."""

    dataset = PhysioChemDataset(
        [sequence], np.array([0.0]),
        config['feature_type'],
        config['max_length']
    )

    features, _ = dataset[0]

    with torch.no_grad():

        if config['feature_type'] == 'physchem':
            inputs = features['physchem'].unsqueeze(0).to(device)
            prediction = model(inputs)
        elif config['feature_type'] == 'onehot':
            inputs = features['sequence'].unsqueeze(0).to(device)
            prediction = model(inputs)
        elif config['feature_type'] == 'both':
            seq_inputs = features['sequence'].unsqueeze(0).to(device)
            physchem_inputs = features['physchem'].unsqueeze(0).to(device)
            prediction = model(seq_inputs, physchem_inputs)

        return prediction.item()


def predict_batch(model, sequences: List[str], config: Dict, device: torch.device,
                 processor: PhysioChemDataProcessor, batch_size: int = 32) -> np.ndarray:
    """Predict properties for a batch of sequences."""
    predictions = []


    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]


        dataset = PhysioChemDataset(
            batch_sequences,
            np.zeros(len(batch_sequences)),
            config['feature_type'],
            config['max_length']
        )


        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn
        )


        batch_predictions = []
        with torch.no_grad():
            for batch_features, _ in dataloader:

                if config['feature_type'] == 'physchem':
                    inputs = batch_features['physchem'].to(device)
                    outputs = model(inputs)
                elif config['feature_type'] == 'onehot':
                    inputs = batch_features['sequence'].to(device)
                    outputs = model(inputs)
                elif config['feature_type'] == 'both':
                    seq_inputs = batch_features['sequence'].to(device)
                    physchem_inputs = batch_features['physchem'].to(device)
                    outputs = model(seq_inputs, physchem_inputs)

                batch_predictions.extend(outputs.cpu().numpy().flatten())

        predictions.extend(batch_predictions)

    return np.array(predictions)


def predict_from_file(model_path: str, input_file: str, output_file: str,
                     sequence_col: str = 'sequence', batch_size: int = 32):
    """Predict properties from input file and save results."""

    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")


    print(f"Loading model from {model_path}...")
    model, config = load_model(model_path, device)
    print(f"Model type: {config['model_type']}, Feature type: {config['feature_type']}")


    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    sequences = df[sequence_col].astype(str).tolist()
    sequences = [seq.strip().upper() for seq in sequences]

    print(f"Found {len(sequences)} sequences")


    processor = PhysioChemDataProcessor(
        feature_type=config['feature_type'],
        max_length=config['max_length']
    )


    print("Making predictions...")
    predictions = predict_batch(model, sequences, config, device, processor, batch_size)


    results_df = df.copy()
    results_df['predicted_property'] = predictions


    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


    print("\nPrediction Statistics:")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  Std: {predictions.std():.4f}")
    print(f"  Min: {predictions.min():.4f}")
    print(f"  Max: {predictions.max():.4f}")


def predict_interactive(model_path: str):
    """Interactive prediction mode."""

    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')


    print(f"Loading model from {model_path}...")
    model, config = load_model(model_path, device)


    processor = PhysioChemDataProcessor(
        feature_type=config['feature_type'],
        max_length=config['max_length']
    )

    print("\nInteractive prediction mode. Enter sequences (or 'quit' to exit):")

    while True:
        sequence = input("\nSequence: ").strip()

        if sequence.lower() == 'quit':
            break

        if not sequence:
            continue

        try:
            prediction = predict_single(model, sequence, config, device, processor)
            print(f"Predicted property: {prediction:.4f}")
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Predict physicochemical properties')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input CSV file with sequences')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output CSV file for predictions')
    parser.add_argument('--sequence_col', type=str, default='sequence',
                       help='Name of sequence column')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for prediction')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')

    args = parser.parse_args()

    if args.interactive:
        predict_interactive(args.model_path)
    elif args.input_file and args.output_file:
        predict_from_file(
            args.model_path, args.input_file, args.output_file,
            args.sequence_col, args.batch_size
        )
    else:
        print("Error: Either provide --input_file and --output_file, or use --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()

