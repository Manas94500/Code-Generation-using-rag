# Code-Generation-using-rag
Retrieval-Augmented Generation (RAG) with LangChain + Mistral-7B
This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline that answers user questions strictly from provided PDF documents. If the answer is not present in the retrieved context, the system explicitly responds with “I don’t know”, preventing hallucinations.

The implementation uses LangChain, FAISS, and a quantized Mistral-7B Instruct model for efficient local inference on consumer GPUs.

🚀 Features

📚 Loads and processes multiple PDF documents

✂️ Chunking with overlap for better semantic retrieval

🔍 Vector search using FAISS

🧠 Context-aware answers using Mistral-7B (4-bit quantized)

❌ Graceful fallback when context is missing

⚡ Optimized for local GPU inference (tested on RTX 4060)

🧠 How It Works

PDF Loading
PDFs are loaded using PyPDFLoader.

Text Splitting
Documents are split into overlapping chunks to preserve context.

Embedding Generation
Each chunk is embedded using sentence-transformers/all-MiniLM-L6-v2.

Vector Storage & Retrieval
FAISS is used to store embeddings and retrieve the top-k relevant chunks.

Prompting Strategy
The model is instructed to:

Answer only using retrieved context

Respond with “I don’t know” if the answer is not found

LLM Inference
Uses a 4-bit quantized Mistral-7B-Instruct model via Hugging Face pipeline.
Example Outputs
Successful Answer from Context
(Image shows correct answer generated strictly from retrieved documents)
<img width="1825" height="436" alt="image" src="https://github.com/user-attachments/assets/9caa1639-2680-4a67-bb4f-9039476206a9" />
❓ Context Not Found Case
(Image shows model responding with “I don’t know” when answer is absent)
<img width="1653" height="141" alt="image" src="https://github.com/user-attachments/assets/51598468-5f0d-423c-ab12-ac5e88896252" />
🧩 Tech Stack

Language: Python

LLM: Mistral-7B-Instruct-v0.2

Framework: LangChain

Vector Store: FAISS

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Quantization: BitsAndBytes (4-bit NF4)

Hardware: NVIDIA GPU (RTX 4060 tested)

⚠️ Limitations

Answers are strictly limited to provided documents

No web search or external knowledge

Performance depends on document quality and chunking strategy

📌 Future Improvements

Add citation highlighting for retrieved chunks

Support for more document formats

Web UI using Streamlit or Gradio

Conditional retrieval for lower latency

📄 License

This project is self-authored and currently does not use an external license.
