from typing import List
import numpy as np
import faiss


# Modified from original HippoRAG repo to use FAISS for KNN retrieval

def retrieve_knn(query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k=2047,
                 query_batch_size=1000, key_batch_size=10000):
    """
    Retrieve the top-k nearest neighbors for each query id from the key ids using FAISS.
    Args:
        query_ids: List of query identifiers
        key_ids: List of key identifiers
        query_vecs: Query vectors (numpy array or list)
        key_vecs: Key vectors (numpy array or list)
        k: top-k neighbors to retrieve
        query_batch_size: (unused, kept for API compatibility)
        key_batch_size: (unused, kept for API compatibility)

    Returns:
        Dictionary mapping query_id -> (list of top-k key_ids, list of similarity scores)
    """
    if len(key_vecs) == 0:
        return {}

    # Convert to numpy arrays and normalize for cosine similarity
    query_vecs = np.array(query_vecs, dtype=np.float32)
    key_vecs = np.array(key_vecs, dtype=np.float32)
    
    # Normalize vectors for cosine similarity (FAISS IndexFlatIP computes inner product)
    faiss.normalize_L2(query_vecs)
    faiss.normalize_L2(key_vecs)

    d = key_vecs.shape[1]  # dimension
    k = min(k, len(key_vecs))  # can't retrieve more than we have

    # Use GPU if available, otherwise CPU
    if faiss.get_num_gpus() > 0:
        # GPU index for faster search
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, d)
    else:
        # CPU index
        index = faiss.IndexFlatIP(d)

    # Add key vectors to the index
    index.add(key_vecs)

    # Search for k nearest neighbors
    sim_scores, indices = index.search(query_vecs, k)

    # Build results dictionary
    results = {}
    for i, query_id in enumerate(query_ids):
        topk_key_ids = [key_ids[idx] for idx in indices[i] if idx >= 0]
        topk_scores = sim_scores[i][:len(topk_key_ids)].tolist()
        results[query_id] = (topk_key_ids, topk_scores)

    return results