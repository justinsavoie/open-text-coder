import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

# 1. Embed
embedder = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("data/data-cps21-40.csv")
texts = df['cps21_imp_iss'].fillna('').tolist()
embeddings = embedder.encode(texts, show_progress_bar=True)

# 2. Reduce dimensions (UMAP is perfect for this)
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
embedding_2d = reducer.fit_transform(embeddings)

# 3. Cluster (HDBSCAN > K-means for survey data)
clusterer = HDBSCAN(
    min_cluster_size=2,  # Minimum responses per cluster
    min_samples=5,
    metric='euclidean'
)
clusters = clusterer.fit_predict(embedding_2d)

# 4. Visualize
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    embedding_2d[:, 0], 
    embedding_2d[:, 1], 
    c=clusters, 
    cmap='Spectral',
    s=5,
    alpha=0.7
)
plt.title('Survey Response Clusters')
plt.colorbar(scatter)
plt.show()

# 5. Describe clusters (most important part!)
df['cluster'] = clusters
for cluster_id in range(clusters.max() + 1):
    print(f"\n--- Cluster {cluster_id} ---")
    cluster_texts = df[df['cluster'] == cluster_id]['cps21_imp_iss'].sample(min(5, sum(df['cluster'] == cluster_id)))
    for text in cluster_texts:
        print(f"- {text[:100]}...")