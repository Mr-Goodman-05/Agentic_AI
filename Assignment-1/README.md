# RAG Assignment — README

## 1. Project overview
This project implements a Retrieval-Augmented Generation (RAG) system that answers user queries using knowledge extracted from a set of text-only PDF documents. The system extracts text, chunks documents with overlap, embeds chunks into dense vectors, stores them in a FAISS vector database, retrieves relevant chunks for a given natural language query, and conditions a generation model on the retrieved context to produce grounded answers.

## 2. Dataset description
- **File type:** PDF only  
- **Content type:** Text only (selectable text)  
- **Source:** Public-domain PDFs (Project Gutenberg). The notebook expects three PDFs in the working directory:
  - `alice_in_wonderland.pdf`
  - `pride_and_prejudice.pdf`
  - `sherlock_adventures.pdf`

These can be replaced by other public/self-created PDFs but must be text-extractable.

## 3. RAG architecture explanation
Pipeline:
1. **PDF collection** → 2. **Text extraction** (pdfplumber) → 3. **Chunking** (character-based with overlap) → 4. **Embedding** (sentence-transformers) → 5. **Indexing** (FAISS IndexFlatIP with normalized vectors) → 6. **Retrieval** (top-k nearest) → 7. **Context assembly + prompt** → 8. **Generation** (OpenAI or local FLAN-T5).

(See the notebook for a textual block diagram.)

## 4. Tools and libraries used
- `pdfplumber` — PDF text extraction
- `sentence-transformers` (`all-MiniLM-L6-v2`) — embeddings
- `faiss-cpu` — vector database / similarity search
- `transformers` — local generation fallback (google/flan-t5-small)
- `torch` — model backend
- `tqdm`, `numpy`, `pickle`, `json` — utilities

## 5. Chunking strategy
- **Chunk size:** 1000 characters  
- **Chunk overlap:** 200 characters  
**Reason:** Keeps semantic units (paragraphs, short sections) intact, reduces lost context across boundaries, and balances embedding dimension/compute. Character chunking is simple and deterministic; for production token-based chunking (with a tokenizer) is often preferred.

## 6. Embedding model used
- **Model:** `all-MiniLM-L6-v2` (SentenceTransformers)  
**Why:** Fast, compact (384-d), high-quality semantic embeddings for retrieval, runs locally on CPU/GPU, no external API required.

## 7. Vector database explanation
- **FAISS** with `IndexFlatIP` + normalized vectors to simulate cosine similarity search. FAISS is efficient, widely used in research/industry, and supports billions of vectors with extensions. We store a parallel metadata list mapping index positions to chunk metadata for attribution.

## 8. How to run the project
1. Place the required PDFs in the notebook directory with the exact filenames:
   - `alice_in_wonderland.pdf`
   - `pride_and_prejudice.pdf`
   - `sherlock_adventures.pdf`
2. Start Jupyter and open the notebook (or run in Google Colab after uploading files).
3. Execute cells top-to-bottom. The main steps:
   - Install dependencies (`pip install ...`)
   - Extract text from PDFs (`pdfplumber`)
   - Chunk text (1000 char / 200 char overlap)
   - Embed chunks (`all-MiniLM-L6-v2`)
   - Build FAISS index (saved to `faiss_index.bin` and metadata in `index_meta.pkl`)
   - Use `answer_query(query)` to run retrieval + generation.
4. If you want better generation quality, set `OPENAI_API_KEY` in your environment and call `answer_query(query, use_openai_if_available=True)`.

## 9. Future Improvements
- **Token-based chunking** using the generator/LLM tokenizer (e.g., tiktoken/BPE for OpenAI) to match LLM context windows precisely.
- **Hybrid retrieval** combining sparse (BM25) + dense (FAISS) for improved recall.
- **Passage scoring and reranking** using a cross-encoder for higher precision.
- **Metadata-aware retrieval** (filter by section, chapter, or date).
- **Store embeddings in a persistent vector DB** with backups (e.g., Milvus, Weaviate, Pinecone) for scale and persistence.
- **Better generator** integration: use higher-capacity LLMs (OpenAI GPT-4 or local Llama-family with sufficient hardware) and chain-of-thought style prompting when allowed.
- **Evaluation suite:** automated metrics (accuracy vs gold answers), hallucination detection, and human-in-the-loop validation.
