import numpy as np
from minsearch import VectorSearch


def build_vector_index(chunks, embedder, batch_size=32):
    vectors = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk['content'] for chunk in batch]
        vectors.append(embedder.encode_batch(texts))

    X = np.vstack(vectors)

    vindex = VectorSearch(keyword_fields=['filename'])
    vindex.fit(X, chunks)
    return vindex
