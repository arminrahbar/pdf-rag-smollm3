### Overview

This project performs three core tasks.

#### 1. PDF ingestion

- Reads PDF files from the `data/` directory  
- Extracts text from each page using `pdfplumber`  
- Cleans and normalizes the text  

#### 2. Chunking and embeddings

- Splits pages into overlapping text chunks for better retrieval  
- Generates embeddings using  
  `sentence-transformers/all-MiniLM-L6-v2`  
- Creates a FAISS similarity index  

#### 3. Querying

- Takes a natural language question  
- Converts it into an embedding  
- Searches the FAISS index  
- Returns the most relevant chunks plus context information  

This repository currently supports retrieval only, but it is designed for future integration with **SmolLM3-3B** for answer generation.

---

### Project structure

```text
pdf-rag-smollm3/
├── data/
│   ├── (your PDFs go here)
│   ├── index.faiss          # generated locally
│   ├── embeddings.npy       # generated locally
│   └── metadata.json        # generated locally
│
├── src/
│   ├── ingest.py            # builds chunks, embeddings, FAISS index
│   ├── query.py             # runs similarity search on the index
│   └── __init__.py
│
├── .gitignore
├── requirements.txt
└── README.md

```


All PDF files and the generated `index.faiss`, `embeddings.npy`, and `metadata.json` files are intentionally excluded from GitHub for copyright and size reasons.

---

### Installation

Follow these steps to run the project locally.

#### 1. Clone the repository

```text
git clone https://github.com/arminrahbar/pdf-rag-smollm3.git
cd pdf-rag-smollm3
```


### 2. Create a virtual environment

Windows (PowerShell):

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```text
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Run the following command to install the required libraries.

```text
pip install pdfplumber sentence-transformers faiss-cpu
```

### Dependency Versions

This project was developed and tested with the following versions:

```text
Python 3.12.6
pdfplumber 0.11.8
sentence-transformers 5.1.2
faiss-cpu 1.13.0
```

### Adding Your Documents

Place your own PDF files inside the `data/` directory:

```text
data/
├── my_paper_1.pdf
├── my_paper_2.pdf
└── ...
```

These files are private and local.  
They will never be uploaded to GitHub.

---

### Build the Index (Ingest PDFs)

Run the ingestion script to process your PDFs:

```text
python -m src.ingest
```

This will:

- Extract text  
- Chunk it into overlapping segments  
- Generate embeddings  
- Build a FAISS index  

Output files created in `data/`:

    index.faiss
    embeddings.npy
    metadata.json

These files are automatically ignored by git.

---

### Query the Index

Ask natural language questions about your documents:

```text
python -m src.query
```