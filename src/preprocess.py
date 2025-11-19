"""
src/preprocess.py
=================
Task 1: Preprocess the movie plots dataset.

👉 Why we need this?
The original dataset (wiki_movie_plots_deduped.csv) is very large and sometimes messy.
We only need a smaller, clean subset to work with.

This script does:
1. Load the raw dataset (handles encoding issues automatically).
2. Detect the "Title" and "Plot" columns.
3. Clean the plot text (remove newlines, extra spaces).
4. Filter out invalid or too short rows.
5. Sample ~300 rows for faster experiments.
6. Save to a smaller CSV file.

Usage:
    python src/preprocess.py --input data/raw/wiki_movie_plots_deduped.csv --output data/processed/subset.csv --n 300
"""

import argparse  # for command line arguments
import pandas as pd  # for CSV reading and processing
import re  # for cleaning text with regex
import os  # for file and folder handling
from typing import Tuple


# ------------------------------
# Helper function: read CSV safely
# ------------------------------
def read_csv_with_fallback(path: str) -> pd.DataFrame:
    """
    Reads a CSV file.
    Tries UTF-8 encoding first, if that fails tries Latin-1.
    This avoids common encoding errors.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        return pd.read_csv(path, low_memory=False)  # try normal UTF-8
    except Exception:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)  # fallback


# ------------------------------
# Helper function: detect Title and Plot columns
# ------------------------------
def detect_title_and_plot_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Finds which columns are the movie Title and Plot automatically.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        (str, str): Title column name, Plot column name.
    """
    cols = df.columns.tolist()
    # Find the first column with "title" in the name
    title_col = next((c for c in cols if 'title' in c.lower()), None)
    # Find the first column with "plot" in the name
    plot_col = next((c for c in cols if 'plot' in c.lower()), None)
    return title_col, plot_col


# ------------------------------
# Helper function: clean plot text
# ------------------------------
def clean_plot_text(s: object) -> str:
    """
    Cleans text of a plot by:
    - removing newlines/tabs
    - collapsing multiple spaces
    - trimming leading/trailing spaces

    Args:
        s (object): Raw plot text (could be string or NaN).

    Returns:
        str: Cleaned plot text.
    """
    if pd.isna(s):  # if missing value
        return ""
    s = str(s)  # ensure it's a string
    s = re.sub(r'[\r\n\t]+', ' ', s)  # replace line breaks/tabs with space
    s = re.sub(r'\s{2,}', ' ', s)  # collapse multiple spaces
    return s.strip()  # trim spaces at start/end


# ------------------------------
# Main function: preprocess dataset
# ------------------------------
def preprocess(input_path: str, output_path: str, n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Loads raw dataset, cleans it, and samples a smaller subset.

    Args:
        input_path (str): Path to raw CSV (wiki_movie_plots_deduped.csv).
        output_path (str): Where to save the cleaned subset.
        n (int): Number of rows to sample.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Processed DataFrame (subset).
    """
    # 1. Load dataset
    df_raw = read_csv_with_fallback(input_path)

    # 2. Find title and plot columns
    title_col, plot_col = detect_title_and_plot_columns(df_raw)
    if not title_col or not plot_col:
        raise ValueError(f"Couldn't find Title/Plot columns. Found: {df_raw.columns.tolist()}")

    # 3. Select only title and plot columns, rename for consistency
    df = df_raw[[title_col, plot_col]].rename(columns={title_col: "title", plot_col: "plot"})

    # 4. Clean data
    df['title'] = df['title'].astype(str).str.strip()  # make sure title is string and clean
    df['plot'] = df['plot'].apply(clean_plot_text)  # clean plots

    # 5. Filter invalid rows
    df = df[df['title'].str.len() > 0]  # remove empty titles
    df = df[df['plot'].str.len() > 20]  # remove very short plots
    df['plot_word_count'] = df['plot'].str.split().apply(len)  # add word count
    df = df[df['plot_word_count'] >= 10]  # filter very tiny plots

    # 6. Sample subset reproducibly
    n_actual = min(n, len(df))
    df_sample = df.sample(n=n_actual, random_state=seed).reset_index(drop=True)

    # 7. Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_sample.to_csv(output_path, index=False)

    return df_sample


# ------------------------------
# Command-line interface (CLI)
# ------------------------------
def cli():
    """
    Allows running the script from the terminal.

    Example:
        python src/preprocess.py --input data/raw/wiki_movie_plots_deduped.csv --output data/processed/subset.csv --n 300
    """
    parser = argparse.ArgumentParser(description="Preprocess the movie plots dataset.")
    parser.add_argument("--input", required=True, help="Path to raw dataset (wiki_movie_plots_deduped.csv)")
    parser.add_argument("--output", required=True, help="Path to save processed subset (subset.csv)")
    parser.add_argument("--n", type=int, default=300, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Run preprocessing
    df = preprocess(args.input, args.output, args.n, args.seed)

    print(f"Saved {len(df)} rows to {args.output}")


# ------------------------------
# Run if script is executed directly
# ------------------------------
if __name__ == "__main__":
    cli()
