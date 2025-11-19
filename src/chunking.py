"""
src/chunking.py
================
Task 2: Chunking the movie plots dataset.

👉 Why we need this?
When we use embeddings (later in Task 3), models have a token/word limit.
If a plot is too long (like 500–1000 words), we must break it into smaller "chunks".
This script splits each plot into chunks of about 300 words, with 50 words overlap.

👉 Example:
If a plot has 600 words, and we use chunk_size=300, overlap=50:
- Chunk 1 = words [0:300]
- Chunk 2 = words [250:550]  (notice 50 overlap with previous)
- Chunk 3 = words [500:600]

This overlap helps preserve context between chunks.
"""

import argparse  # for command line arguments
import pandas as pd  # for working with CSV files easily
import os  # for saving files and creating folders


# ------------------------------
# Helper function: split text into chunks
# ------------------------------
def chunk_text(words, chunk_size=300, overlap=50):
    """
    Splits a list of words into overlapping chunks.

    Args:
        words (list[str]): The plot text already split into words.
        chunk_size (int): Max number of words per chunk.
        overlap (int): Number of words to overlap between chunks.

    Returns:
        list[str]: A list of chunk strings.
    """
    chunks = []
    start = 0  # where we begin slicing words

    while start < len(words):  # repeat until we reach the end of the plot
        end = min(start + chunk_size, len(words))  # don't go past the end
        chunk_words = words[start:end]  # take words from start to end
        chunks.append(" ".join(chunk_words))  # join words back into a string

        if end == len(words):  # if we reached the end of the plot, stop
            break

        # move the start forward, keeping some overlap
        start += chunk_size - overlap

    return chunks


# ------------------------------
# Main function: chunk all plots in the dataset
# ------------------------------
def chunk_plots(input_path: str, output_path: str, chunk_size: int = 300, overlap: int = 50):
    """
    Loads the processed subset of movie plots and splits each plot into chunks.

    Args:
        input_path (str): Path to input CSV (e.g., data/processed/subset.csv).
        output_path (str): Path to save chunked dataset (e.g., data/processed/chunks.csv).
        chunk_size (int): Max words per chunk.
        overlap (int): Overlap between chunks.

    Returns:
        pd.DataFrame: DataFrame of chunked plots with metadata.
    """
    # Load the subset file (output from Task 1)
    df = pd.read_csv(input_path)

    all_chunks = []  # will hold dictionaries of chunk data

    # Loop through each row (movie) in the dataset
    for _, row in df.iterrows():
        title = row["title"]  # movie title
        plot = row["plot"]  # full cleaned plot
        words = str(plot).split()  # split into list of words

        # Split this plot into chunks
        chunks = chunk_text(words, chunk_size=chunk_size, overlap=overlap)

        # For each chunk, store metadata + chunk text
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "title": title,  # which movie
                "chunk_id": i,  # 0, 1, 2... for each chunk
                "chunk_text": chunk,  # the actual text of the chunk
                "original_plot_word_count": len(words)  # how long the original plot was
            })

    # Convert list of dictionaries into a DataFrame
    df_chunks = pd.DataFrame(all_chunks)

    # Make sure output folder exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save to CSV
    df_chunks.to_csv(output_path, index=False)

    return df_chunks


# ------------------------------
# Command-line interface (CLI)
# ------------------------------
def cli():
    """
    Allows the script to be run from the command line.
    Example:
        python src/chunking.py --input data/processed/subset.csv --output data/processed/chunks.csv --chunk_size 300 --overlap 50
    """
    parser = argparse.ArgumentParser(description="Chunk movie plots into smaller pieces.")
    parser.add_argument("--input", required=True, help="Path to input CSV (subset.csv)")
    parser.add_argument("--output", required=True, help="Path to save output CSV (chunks.csv)")
    parser.add_argument("--chunk_size", type=int, default=300, help="Max words per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap words between chunks")
    args = parser.parse_args()

    # Run the chunking process
    df_chunks = chunk_plots(args.input, args.output, args.chunk_size, args.overlap)

    print(f"Created {len(df_chunks)} chunks. Saved to {args.output}")


# ------------------------------
# Run if script is executed directly
# ------------------------------
if __name__ == "__main__":
    cli()
