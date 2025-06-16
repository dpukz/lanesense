from huggingface_hub import hf_hub_download
import shutil, os

# --- 1. Download supercombo.onnx from the CommaAI repo ---
model_path = hf_hub_download(
    repo_id="CommaAI/supercombo",
    filename="supercombo.onnx",
    cache_dir=os.getcwd(),          # put it right here
    library_name="hf_hub_download_demo",
    library_version="1.0"
)

# --- 2. Move it (if it ended up in a subfolder) ---
dst = os.path.join(os.getcwd(), "supercombo.onnx")
if model_path != dst:
    shutil.move(model_path, dst)

print("✅ Downloaded model to", dst)
