import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel

vertexai.init(project="agent-assistant-3000", location="us-central1")

print("Testing model access...\n")

# Test Gemini models
gemini_models = [
    "gemini-1.5-flash-002",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro-002",
    "gemini-pro",
]

print("Gemini models:")
for model_name in gemini_models:
    try:
        model = GenerativeModel(model_name)
        # Try a simple generation to confirm it works
        response = model.generate_content("Say 'test'")
        print(f"  ✓ {model_name} - WORKS")
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"  ✗ {model_name} - FAILED: {error_msg}")

# Test PaLM 2 models
print("\nPaLM 2 text generation models:")
palm_models = [
    "text-bison@002",
    "text-bison@001",
    "text-bison",
]

for model_name in palm_models:
    try:
        model = TextGenerationModel.from_pretrained(model_name)
        response = model.predict("test", max_output_tokens=10)
        print(f"  ✓ {model_name} - WORKS")
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"  ✗ {model_name} - FAILED: {error_msg}")

# Test embedding models
print("\nEmbedding models:")
embedding_models = [
    "text-embedding-004",
    "textembedding-gecko@003",
    "textembedding-gecko@002",
    "textembedding-gecko@001",
]

for model_name in embedding_models:
    try:
        model = TextEmbeddingModel.from_pretrained(model_name)
        embeddings = model.get_embeddings(["test"])
        print(f"  ✓ {model_name} - WORKS")
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"  ✗ {model_name} - FAILED: {error_msg}")

print("\n" + "="*50)
print("Run this script to find which models work in your project!")