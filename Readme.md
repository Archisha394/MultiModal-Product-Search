# Multi-Modal AI Product Search & Recommendation

## Features
- Semantic text search using SentenceTransformers
- Image search using OpenAI CLIP
- Hybrid capability ready
- FAISS for vector similarity
- Backend: FastAPI
- Frontend: Gradio or React


# Setup Environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# create embeddings
python data_prep.py 

# Build FAISS index
python build_index.py 

## Run Streamlit
python app.py

