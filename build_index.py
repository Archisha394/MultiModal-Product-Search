import argparse
import faiss
import numpy as np
import os

def normalize_embeddings(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    return x / norms

def main(args):
    assert os.path.exists(args.text_embeddings), "Text embeddings file not found"
    assert os.path.exists(args.image_embeddings), "Image embeddings file not found"

    text_emb = np.load(args.text_embeddings).astype('float32')
    img_emb = np.load(args.image_embeddings).astype('float32')

    print(f"Loaded text embeddings: {text_emb.shape}")
    print(f"Loaded image embeddings: {img_emb.shape}")

    # Combine into one matrix
    x = np.vstack([text_emb, img_emb])
    x = normalize_embeddings(x)

    d = x.shape[1]
    print("Creating IndexFlatIP...")
    index = faiss.IndexFlatIP(d)
    index.add(x)
    print("Index size:", index.ntotal)

    faiss.write_index(index, args.out_index)
    print(f"Wrote index to {args.out_index}")

    meta = {
        'n': x.shape[0],
        'dim': d,
        'text_count': text_emb.shape[0],
        'image_count': img_emb.shape[0]
    }
    np.save(args.meta_out, meta)
    print(f"Saved meta to {args.meta_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_embeddings', type=str, default='embeddings/text_embeddings.npy')
    parser.add_argument('--image_embeddings', type=str, default='embeddings/image_embeddings.npy')
    parser.add_argument('--out_index', type=str, default='faiss_index.faiss')
    parser.add_argument('--meta_out', type=str, default='index_meta.npy')
    args = parser.parse_args()
    main(args)
