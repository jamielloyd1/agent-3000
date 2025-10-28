import os
import numpy as np
import streamlit as st
import google.generativeai as genai
import vertexai
from vertexai.language_models import TextEmbeddingModel
from dotenv import load_dotenv

# ===============================
# 0. Configure APIs
# ===============================
# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# For embeddings, still use Vertex AI (since it works)
vertexai.init(project="agent-assistant-3000", location="us-central1")

# ===============================
# 1. Functions
# ===============================

def collect_readmes(root_folder):
    """Recursively collect all README files from root_folder."""
    readme_texts = {}
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().startswith("readme") and filename.lower().endswith((".md", ".txt")):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    readme_texts[filepath] = f.read()
    return readme_texts

def chunk_text(text, max_words=500):
    """Split long text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def create_embeddings(readme_texts):
    """Generate embeddings for all README chunks."""
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    embeddings = {}
    for path, text in readme_texts.items():
        chunks = chunk_text(text)
        embeddings[path] = [emb.values for emb in model.get_embeddings(chunks)]
    return embeddings

def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(user_input, embeddings):
    """Find the README path with the most similar content to the user input."""
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    user_embedding = model.get_embeddings([user_input])[0].values

    best_score = -1
    best_path = None
    for path, chunk_embeddings in embeddings.items():
        for chunk_embedding in chunk_embeddings:
            score = cosine_sim(user_embedding, chunk_embedding)
            if score > best_score:
                best_score = score
                best_path = path
    return best_path, best_score

def generate_suggestion(user_input, readme_text):
    """Generate a human-readable agent suggestion using Google AI."""
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    
    prompt = f"""User wants to build: {user_input}

Here is a README from a similar project:
{readme_text[:3000]}

Based on this, suggest an agent the user could build. Be concise and actionable.
"""
    
    response = model.generate_content(prompt)
    return response.text

# ===============================
# 2. Streamlit UI
# ===============================

st.title("AI Agent Suggester")
st.write(
    "Type in what kind of agent you want to build, and this tool will suggest a relevant agent based on your existing README files."
)

ROOT_FOLDER = "Data/all-readmes"

# Load READMEs and embeddings once, cached for speed
@st.cache_data
def load_data():
    readmes = collect_readmes(ROOT_FOLDER)
    embeddings = create_embeddings(readmes)
    return readmes, embeddings

try:
    readmes, embeddings = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Make sure the 'Data/all-readmes' folder exists and contains README files")
    st.stop()

# User input
user_input = st.text_input("Describe your agent:")

if user_input:
    best_path, score = find_best_match(user_input, embeddings)
    st.write(f"**Closest README match:** {best_path} (score: {score:.3f})")
    
    try:
        suggestion = generate_suggestion(user_input, readmes[best_path])
        st.write("**Suggested agent:**")
        st.write(suggestion)
    except Exception as e:
        st.error(f"Error generating suggestion: {str(e)}")
        st.info("Try running 'python list_models.py' to see available models")