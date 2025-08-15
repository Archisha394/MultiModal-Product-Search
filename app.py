import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import utils # Your utils.py file
import faiss

# --- Configuration ---
st.set_page_config(page_title="👗 Multimodal Fashion Search", layout="wide")

MODEL_NAME = 'clip-ViT-B-32'
INDEX_FILE = 'faiss_index.faiss'
INDEX_META_FILE = 'index_meta.npy'
METADATA_FILE = 'data/metadata_out.csv' # The output from data_prep.py

# --- Load Resources (with caching) ---
@st.cache_resource
def load_all_resources():
    """Load all necessary models, indexes, and metadata."""
    print("Loading resources...")
    model = utils.load_model(MODEL_NAME)
    index = utils.load_faiss_index(INDEX_FILE)
    metadata_df = utils.load_metadata(METADATA_FILE)
    index_meta = np.load(INDEX_META_FILE, allow_pickle=True).item()
    text_count = index_meta['text_count']
    print("Resources loaded successfully.")
    return model, index, metadata_df, text_count

model, index, df, TEXT_COUNT = load_all_resources()

# --- Core Search Logic ---
def search(query_vector, top_k=9):
    """
    Searches the FAISS index and maps results back to the original metadata.
    The index contains text embeddings first, then image embeddings.
    So, result index `i` corresponds to metadata row `i % TEXT_COUNT`.
    """
    # Normalize the query vector for IndexFlatIP
    query_vector = query_vector / np.linalg.norm(query_vector)
    query_vector = np.expand_dims(query_vector, axis=0) # Reshape for FAISS

    # Search the index
    distances, indices = utils.search_index(index, query_vector, top_k=top_k)

    # Process results
    results = []
    # indices[0] contains the indices of the top_k results
    for i in indices[0]:
        # The key logic: map the index from the combined (text+image) index
        # back to the original item index in the metadata dataframe.
        item_index = i % TEXT_COUNT
        results.append(df.iloc[item_index])

    # Remove duplicates because a query might match both the text and image
    # embedding of the same item.
    unique_results_df = pd.DataFrame(results).drop_duplicates(subset=['index'])
    return unique_results_df

def get_styling_recommendations(query_embedding, top_k=6):
    """
    Finds items from complementary categories to style with the input item.
    This version relies only on masterCategory, as subCategory and articleType
    are not in the default metadata_out.csv.
    """
    # 1. Find the category of the input item by finding its nearest neighbor
    D, I = index.search(np.expand_dims(query_embedding, axis=0).astype('float32'), 1)
    input_item_index = I[0][0] % TEXT_COUNT
    input_item_row = df.iloc[input_item_index]
    input_master_category = input_item_row['masterCategory']

    # 2. Define styling rules based on MasterCategory
    styling_rules = {
        "Topwear": {"master": ["Bottomwear", "Footwear"]},
        "Bottomwear": {"master": ["Topwear", "Footwear"]},
        "Footwear": {"master": ["Apparel", "Bottomwear"]},
        "Apparel": {"master": ["Footwear", "Accessories"]},
        "Accessories": {"master": ["Apparel"]},
        "Saree": {"master": ["Accessories", "Footwear"]},
        "Dress": {"master": ["Footwear", "Accessories"]},
        "Bags": {"master": ["Apparel"]}
    }

    # 3. Determine which rule to use
    target_masters = []
    if input_master_category in styling_rules:
        st.info(f"Finding items to style with your **{input_master_category}** item...")
        rule = styling_rules[input_master_category]
        target_masters = rule.get("master", [])
    else:
        st.warning(f"No specific styling rules for **{input_master_category}**. Showing generally similar items as a fallback.")
        return search(query_embedding, top_k)

    # 4. Filter metadata for items in target categories
    if not target_masters:
        st.warning("Could not determine complementary categories. Showing similar items.")
        return search(query_embedding, top_k)
        
    master_mask = df['masterCategory'].isin(target_masters)
    recommendation_pool_df = df[master_mask]

    if recommendation_pool_df.empty:
        st.warning("Could not find any complementary items. Showing similar items instead.")
        return search(query_embedding, top_k)

    pool_indices = recommendation_pool_df['index'].values.astype('int64')

    # 5. Reconstruct the image vectors for these items from the main index
    pool_image_vectors = np.array([index.reconstruct(int(i + TEXT_COUNT)) for i in pool_indices])
    pool_image_vectors = utils.normalize(pool_image_vectors)

    # 6. Build a temporary FAISS index for the recommendation pool
    d = pool_image_vectors.shape[1]
    rec_index = faiss.IndexFlatIP(d)
    rec_index = faiss.IndexIDMap(rec_index)
    rec_index.add_with_ids(pool_image_vectors.astype('float32'), pool_indices)

    # 7. Search this temporary index
    query_vector = np.expand_dims(query_embedding, axis=0)
    distances, rec_indices = rec_index.search(query_vector.astype('float32'), top_k)

    # 8. Get final results from the main dataframe
    final_indices = rec_indices[0]
    final_indices = [i for i in final_indices if i != -1]
    results = df[df['index'].isin(final_indices)]

    return results


def show_results(results_df):
    """Displays search results in a grid."""
    if results_df.empty:
        st.warning("No results found.")
        return

    st.subheader("Search Results")
    num_cols = min(len(results_df), 3)
    cols = st.columns(num_cols)

    for i, row in enumerate(results_df.itertuples()):
        with cols[i % num_cols]:
            if os.path.exists(row.filepath):
                st.image(row.filepath, use_column_width=True)
                st.markdown(f"**{row.text_to_embed}**")
                st.caption(f"Category: {row.masterCategory}")
            else:
                st.error(f"Image not found:\n{row.filepath}")

# --- Streamlit UI ---
st.title("👗 Multimodal Fashion Search Engine")
st.markdown("Search for fashion products using text, an image, or a combination of both.")

# --- Sidebar for Search Mode Selection ---
st.sidebar.title("Search Options")
search_mode = st.sidebar.radio(
    "Choose your search mode:",
    ("Text Search", "Image Search", "Hybrid Search (Text + Image)", "Styling Helper")
)

# --- Main Page Layout ---

if search_mode == "Text Search":
    st.header("Search with Text")
    text_query = st.text_input("Enter your search query (e.g., 'black leather jacket')", "red floral dress")
    if st.button("Search"):
        if text_query:
            with st.spinner("Searching for products..."):
                query_vector = utils.encode_text(model, text_query)
                results = search(query_vector)
                show_results(results)
        else:
            st.warning("Please enter a text query.")

elif search_mode == "Image Search":
    st.header("Search with an Image")
    uploaded_file = st.file_uploader("Upload an image to find similar products", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Your Uploaded Image', width=200)

        if st.button("Find Similar Products"):
            with st.spinner("Analyzing image and searching..."):
                query_vector = utils.encode_image(model, image)
                results = search(query_vector)
                show_results(results)

elif search_mode == "Hybrid Search (Text + Image)":
    st.header("Hybrid Search: Refine Image with Text")
    text_query = st.text_input("Describe what you want to change or find (e.g., 'in blue', 'without sleeves')", "in a different color")
    uploaded_file = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])

    col1, col2 = st.columns(2)
    with col1:
        st.write("Reference Image:")
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, width=200)
    with col2:
        st.write("Text Modifier:")
        st.info(f"Find items like the image, but **'{text_query}'**")

    # Sliders for weighting
    st.sidebar.subheader("Hybrid Search Weights")
    image_weight = st.sidebar.slider("Image Importance", 0.0, 1.0, 0.5, 0.1)
    text_weight = 1.0 - image_weight
    st.sidebar.write(f"Text Importance: {text_weight:.2f}")


    if st.button("Search"):
        if uploaded_file and text_query:
            with st.spinner("Performing hybrid search..."):
                # Encode both modalities
                image_vector = utils.encode_image(model, image)
                text_vector = utils.encode_text(model, text_query)

                # Combine them using the weights
                hybrid_vector = (image_weight * image_vector) + (text_weight * text_vector)

                results = search(hybrid_vector)
                show_results(results)
        else:
            st.warning("Please provide both an image and a text query for hybrid search.")

elif search_mode == "Styling Helper":
    st.header("Styling Helper")
    st.markdown("Upload an image of a single clothing item to get recommendations for a complete outfit.")
    uploaded_file = st.file_uploader("Upload an item to style", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Your Item', width=200)

        if st.button("Get Styling Options"):
            with st.spinner("Putting together some outfit ideas..."):
                query_vector = utils.encode_image(model, image)
                results = get_styling_recommendations(query_vector)
                show_results(results)
