# Retrieval-Augmented Generation (RAG) with Wiki Movie Plots

This project is a simple end-to-end demo of a **Retrieval-Augmented Generation (RAG)** system built on top of the Wiki Movie Plots dataset.  

The idea is straightforward:  
1. Take the movie plots dataset.  
2. Preprocess and split into smaller text chunks.  
3. Turn those chunks into embeddings and store them in a vector database (Chroma).  
4. Ask a question → retrieve the most relevant chunks.  
5. Use a local Hugging Face model (Flan-T5) to generate an answer grounded in the retrieved text.  
6. Provide both a command-line interface and a small Gradio web app to try it out.  
7. Add an evaluation script to test multiple queries at once.  

---

## Why this project?

- To show how you can build a working RAG pipeline locally, without API keys.  
- To help beginners understand each step of a retrieval-augmented workflow.  
- To provide clean, well-commented starter code that can be extended or swapped with other datasets/models.  

---

## How to run it

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess the dataset**
   ```bash
   python src/preprocess.py --input data/raw/wiki_movie_plots_deduped.csv --output data/processed/subset.csv --n 300
   ```

3. **Chunk the text**
   ```bash
   python src/chunking.py --input data/processed/subset.csv --output data/processed/chunks.csv --chunk_size 300 --overlap 50
   ```

4. **Create embeddings with Chroma**
   ```bash
   python src\embeddings.py --input data\processed\chunks.csv --persist_dir data\vectorstore --collection movie_chunks --batch_size 128
   ```

5. **Try retrieval**
   ```bash
   python src\retrieve.py --query "a ship hits an iceberg" --persist_dir data\vectorstore --collection movie_chunks --k 3 --return_json

   ```

6. **Generate an answer**
   ```bash
   python src\generate.py --query "a ship hits an iceberg" --persist_dir data\vectorstore --collection movie_chunks --k 3 --model_name google/flan-t5-small --max_input_tokens 1024 --max_new_tokens 256 --temperature 0.2 --top_p 0.9 --save_json data\outputs\answer_iceberg.json
   ```

7. **Run the web app**
   ```bash
   python src/app.py --persist_dir data/vectorstore --model_name google/flan-t5-base
   ```
   Then open the URL shown in the terminal (usually http://127.0.0.1:7860).

8. **Batch evaluate multiple queries**
   ```bash
   python src\evaluate.py --persist_dir data\vectorstore --collection movie_chunks --model_name google/flan-t5-base --k 3 --max_new_tokens 128 --deterministic true --out data\outputs\eval_results.json
   ```

---

## Example Queries

- “a ship hits an iceberg” → should return *Atlantic*  
- “a German U-boat battle” → should return *The Enemy Below*  
- “purple dragons invade mars” → should return *I don’t know from the provided context.*  

---

## Notes

- Works on **CPU**. A GPU speeds things up but isn’t required.  
- Uses open Hugging Face models: `flan-t5-small` (fast) or `flan-t5-base` (better).  
- No API keys are needed. Everything runs locally.  
- Chroma stores embeddings in `data/vectorstore/`. If you want to reset, just delete that folder and re-run embeddings.  

---

## Project Structure

```
data/            # dataset, processed CSVs, outputs, vectorstore
src/             # all Python scripts (preprocess, chunk, embeddings, retrieve, generate, app, evaluate)
requirements.txt
```

---

## License

MIT
