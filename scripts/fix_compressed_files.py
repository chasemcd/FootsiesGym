import glob
import gzip
import json
import os

from tqdm import tqdm

DATA_NAME = "test-trajectory-export"
INPUT_DIR = f"FootsiesTrajectories/{DATA_NAME}"
OUTPUT_DIR = f"FootsiesTrajectories/{DATA_NAME}_fixed"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all compressed files
files = glob.glob(f"{INPUT_DIR}/*.json.gz")

for file_path in tqdm(files, desc="Fixing files"):
    # Read the double-compressed file
    with open(file_path, "rb") as f:
        compressed_data = f.read()

    # Decompress twice
    decompressed_once = gzip.decompress(compressed_data)
    json_data = gzip.decompress(decompressed_once)

    # Parse the JSON
    trajectories = json.loads(json_data.decode("utf-8"))

    # Write to new file with correct compression
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(file_path))
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(trajectories, f)

print(f"Fixed files saved to {OUTPUT_DIR}")
