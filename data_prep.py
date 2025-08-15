"""
data_prep.py - Updated for your dataset columns
- Scans images folder (recursively)
- Reads metadata CSV with columns:
  id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName
- Generates:
    - image embeddings (.npy)
    - text embeddings (.npy)
    - cleaned metadata CSV with index, filepath, text_to_embed
- CPU-friendly batching
"""

import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

def list_images(img_dir):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = []
    for root, _, filenames in os.walk(img_dir):
        for f in filenames:
            if os.path.splitext(f.lower())[1] in exts:
                files.append(os.path.join(root, f))
    files.sort()
    return files

def fix_csv_rows(meta_path, expected_columns=10):
    """Fix CSV rows with extra commas in productDisplayName"""
    fixed_lines = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
                fixed_lines.append(line)  # header
            else:
                parts = line.split(",")
                if len(parts) > expected_columns:
                    first9 = parts[:9]
                    product_name = ",".join(parts[9:])
                    fixed_lines.append(",".join(first9 + [product_name]))
                else:
                    fixed_lines.append(line)
    # Save temporary fixed CSV
    temp_path = "metadata_fixed.csv"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(fixed_lines))
    return temp_path

def load_metadata(meta_path):
    if meta_path and os.path.exists(meta_path):
        return pd.read_csv(meta_path)
    return None

# Later when saving metadata
#pd.DataFrame(rows).to_csv(args.out_metadata, index=False, quoting=pd.io.common.csv.QUOTE_ALL)


def image_to_pil(path, target_size=None):
    img = Image.open(path).convert('RGB')
    if target_size:
        img = img.resize(target_size, Image.BICUBIC)
    return img

def main(args):
    device = 'cpu'
    print("Loading model (CPU)...")
    model = SentenceTransformer(args.model_name, device=device)
    model.max_seq_length = 128

    image_files = list_images(args.image_dir)
    print(f"Found {len(image_files)} images under {args.image_dir}")

    metadata_df = load_metadata(args.metadata_csv)
    rows = []

    batch_size = args.batch_size
    image_embeddings = []
    text_embeddings = []

    for i in tqdm(range(0, len(image_files), batch_size), desc="Embedding batches"):
        batch_files = image_files[i:i+batch_size]
        imgs = [image_to_pil(p) for p in batch_files]

        # Image embeddings
        with torch.no_grad():
            img_embs = model.encode(imgs, batch_size=len(imgs), show_progress_bar=False, convert_to_numpy=True)
        image_embeddings.append(img_embs)

        # Text embeddings
        text_batch = []
        for p in batch_files:
            filename = os.path.basename(p)
            text_to_embed = filename  # fallback
            category = ''
            if metadata_df is not None:
                match = metadata_df[metadata_df['id'].astype(str) == os.path.splitext(filename)[0]]
                if len(match) > 0:
                    row_meta = match.iloc[0]
                    text_to_embed = f"{row_meta['productDisplayName']} {row_meta['baseColour']} {row_meta['articleType']} {row_meta['masterCategory']}"
                    category = row_meta['masterCategory']
            text_batch.append(text_to_embed)
            # Save metadata row
            rows.append({
                'index': len(rows),
                'filepath': p,
                'filename': filename,
                'text_to_embed': text_to_embed,
                'masterCategory': category
            })

        with torch.no_grad():
            txt_embs = model.encode(text_batch, batch_size=len(text_batch), show_progress_bar=False, convert_to_numpy=True)
        text_embeddings.append(txt_embs)

        # checkpoint save
        if args.checkpoint_every and (i // batch_size) % args.checkpoint_every == 0:
            np.save(args.out_image_embeddings + '.tmp', np.vstack(image_embeddings))
            np.save(args.out_text_embeddings + '.tmp', np.vstack(text_embeddings))
            pd.DataFrame(rows).to_csv(args.out_metadata + '.tmp', index=False)

    # save final embeddings
    image_embeddings = np.vstack(image_embeddings).astype('float32')
    text_embeddings = np.vstack(text_embeddings).astype('float32')
    np.save(args.out_image_embeddings, image_embeddings)
    np.save(args.out_text_embeddings, text_embeddings)
    pd.DataFrame(rows).to_csv(args.out_metadata, index=False)
    print(f"Saved image embeddings: {args.out_image_embeddings}")
    print(f"Saved text embeddings: {args.out_text_embeddings}")
    print(f"Saved metadata: {args.out_metadata}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/images', help='Folder with images')
    parser.add_argument('--metadata_csv', type=str, default='data/styles.csv', help='CSV with product metadata')
    parser.add_argument('--model_name', type=str, default='clip-ViT-B-32', help='SentenceTransformer model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_image_embeddings', type=str, default='embeddings/image_embeddings.npy')
    parser.add_argument('--out_text_embeddings', type=str, default='embeddings/text_embeddings.npy')
    parser.add_argument('--out_metadata', type=str, default='data/metadata_out.csv')
    parser.add_argument('--checkpoint_every', type=int, default=50)
    args = parser.parse_args()
    main(args)
