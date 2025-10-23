import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Configuración ---
INDEX_PATH = "./faiss_index.index"
METADATA_PATH = "./faiss_metadata.json"

# Modelo de embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Modelo de chat Hugging Face
chat_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# --- Funciones auxiliares ---
def get_embedding(text: str) -> np.ndarray:
    emb = embedding_model.encode([text], convert_to_numpy=True)[0]
    return np.array(emb, dtype="float32")

# --- Cargar FAISS y metadata ---
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# --- Pregunta de prueba ---
user_message = "¿Qué es la violencia intrafamiliar?"

# Embedding y búsqueda
query_embedding = get_embedding(user_message).reshape(1, -1)
distances, indices = index.search(query_embedding, k=3)
retrieved_docs = [metadata[i]["text"] for i in indices[0] if i < len(metadata)]

# Construir prompt
context = "\n\n".join(retrieved_docs)
prompt = f"""
Eres un asistente empático y profesional especializado en orientar a personas sobre violencia intrafamiliar.
Usa el siguiente contexto para responder de forma clara, natural y con sensibilidad.

Contexto:
{context}

Pregunta del usuario:
{user_message}

Respuesta:
"""

# Generar respuesta
response = chat_pipeline(prompt, max_new_tokens=300, temperature=0.4, do_sample=True)
answer = response[0]["generated_text"]

print("=== RESPUESTA FINAL ===")
print(answer)