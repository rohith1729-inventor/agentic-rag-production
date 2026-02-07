
# Download script to bake model into image
import os
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Pre-downloading model: {model_name}")
model = SentenceTransformer(model_name)
save_path = os.getenv("MODEL_PATH", "./models/all-MiniLM-L6-v2")
model.save(save_path)
print(f"Model saved to {save_path}")
