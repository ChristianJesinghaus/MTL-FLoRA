#!/usr/bin/env python3
"""
Script to extract and summarize TinyLlama mLoRA evaluation results.

This script scans the `logs` directory for TinyLlama training and evaluation
output files, extracts the final GLUE average score from each run, and
produces CSV summaries that map each result back to its parameter
configuration. It also identifies the top four runs for each strategy
(centralized, federated, and fedit) and writes them to separate files.

Usage:
    python analyze_tinyllama_results.py

Assumes that this script is placed at the repository root, and that
all `.out` files are located under a `logs/` subdirectory. The
file naming convention must follow the patterns described in the project,
e.g.:

    tinyllama_train_centralized_epoch3_flround1_numB2.out
    tinyllama_eval_federated_epoch1_flround3_numB3.out
    tinyllama_eval_fedit_epoch2_flround2_numB3.out

The script writes three CSV files into the current working directory:

    - results_all.csv: all runs with their parameters and GLUE average
    - top4_centralized.csv: top 4 centralized runs
    - top4_federated.csv: top 4 federated runs
    - top4_fedit.csv: top 4 fedit runs

"""

import re
import csv
from pathlib import Path
from typing import Optional, List, Dict


def extract_glue_avg(file_path: Path) -> Optional[float]:
    """Return the last GLUE average score from a log file, if present.

    Parameters
    ----------
    file_path : Path
        Path to the log file to parse.

    Returns
    -------
    Optional[float]
        The GLUE average score (as a float) rounded to 4 decimal places,
        or None if not found.
    """
    glue_val = None
    # Regex to capture the "avg" value in a line like:
    #   "glue_avg": { "avg": 0.8270597229717576 }
    pattern = re.compile(r'"glue_avg"\s*:\s*\{\s*"avg"\s*:\s*([0-9\.]+)')
    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    try:
                        glue_val = float(match.group(1))
                    except ValueError:
                        continue
    except FileNotFoundError:
        return None
    return round(glue_val, 4) if glue_val is not None else None


def parse_filename(file_name: str) -> Optional[Dict[str, str]]:
    """Parse the filename to extract run parameters.

    Supported patterns:
        tinyllama_train_centralized_epoch3_flround1_numB2.out
        tinyllama_eval_federated_epoch1_flround3_numB3.out
        tinyllama_eval_fedit_epoch2_flround2_numB3.out

    Returns a dict with keys:
        'type' : 'train' or 'eval'
        'strat' : 'centralized', 'federated', or 'fedit'
        'epochs' : str
        'flrounds' : str
        'num_B' : str

    Returns None if the pattern does not match.
    """
    regex = re.compile(
        r'^tinyllama_(train|eval)_(centralized|fedit|federated)_epoch(\d+)_flround(\d+)_numB(\d+)\.out$'
    )
    m = regex.match(file_name)
    if not m:
        return None
    return {
        'type': m.group(1),
        'strat': m.group(2),
        'epochs': m.group(3),
        'flrounds': m.group(4),
        'num_B': m.group(5),
    }


def collect_results(logs_dir: Path) -> List[Dict[str, str]]:
    """Collect GLUE averages from all relevant log files.

    Parameters
    ----------
    logs_dir : Path
        Directory containing the .out log files.

    Returns
    -------
    List[Dict[str, str]]
        A list of dicts, each representing a run with parameters and glue_avg.
    """
    results = []
    for file_path in logs_dir.glob('tinyllama_*.out'):
        meta = parse_filename(file_path.name)
        if not meta:
            # skip files that do not match expected patterns
            continue
        # Determine which logs to process:
        # For centralized strategy, only training logs contain final evaluation.
        # For federated and fedit strategies, only evaluation logs are considered.
        if meta['strat'] == 'centralized':
            if meta['type'] != 'train':
                # skip evaluation logs for centralized, if any
                continue
        else:
            # Only process evaluation logs for federated and fedit
            if meta['type'] != 'eval':
                continue
        glue = extract_glue_avg(file_path)
        if glue is None:
            # Skip runs without glue_avg found
            continue
        result = {
            'strategy': meta['strat'],
            'epochs': int(meta['epochs']),
            'flrounds': int(meta['flrounds']),
            'num_B': int(meta['num_B']),
            'glue_avg': glue,
            'file': str(file_path.relative_to(logs_dir)),
        }
        results.append(result)
    return results


def write_results(results: List[Dict[str, str]], output_path: Path) -> None:
    """Write all results to a CSV file.

    Parameters
    ----------
    results : list of dict
        The run results.
    output_path : Path
        Path to the output CSV file.
    """
    fieldnames = ['strategy', 'epochs', 'flrounds', 'num_B', 'glue_avg', 'file']
    with output_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def write_top_results(results: List[Dict[str, str]], strategy: str, top_n: int, output_path: Path) -> None:
    """Write the top N results for a given strategy to a CSV file.

    Parameters
    ----------
    results : list of dict
        All run results.
    strategy : str
        Strategy to filter on ('centralized', 'federated', 'fedit').
    top_n : int
        Number of top results to write.
    output_path : Path
        Path to the output CSV file.
    """
    # Filter by strategy
    filtered = [r for r in results if r['strategy'] == strategy]
    # Sort by glue_avg descending
    filtered.sort(key=lambda x: x['glue_avg'], reverse=True)
    # Take top N
    top_results = filtered[:top_n]
    fieldnames = ['strategy', 'epochs', 'flrounds', 'num_B', 'glue_avg', 'file']
    with output_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in top_results:
            writer.writerow(row)


def main() -> None:
    logs_dir = Path('logs')
    if not logs_dir.is_dir():
        raise RuntimeError(
            f"Logs directory '{logs_dir}' does not exist. Ensure you're running from the project root."
        )

    # Ergebnisse aus den Logfiles sammeln
    results = collect_results(logs_dir)
    if not results:
        print("No results found. Ensure log files are present in 'logs' directory.")
        return

    # Unterordner für die Ausgabedateien festlegen
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)  # Ordner anlegen, falls er nicht existiert

    # Alle Ergebnisse speichern
    write_results(results, output_dir / 'results_all.csv')

    # Top 4 für jede Strategie speichern
    write_top_results(results, 'centralized', 4, output_dir / 'top4_centralized.csv')
    write_top_results(results, 'federated', 4, output_dir / 'top4_federated.csv')
    write_top_results(results, 'fedit', 4, output_dir / 'top4_fedit.csv')

    print(f"Results saved in '{output_dir}' directory.")


if __name__ == '__main__':
    main()