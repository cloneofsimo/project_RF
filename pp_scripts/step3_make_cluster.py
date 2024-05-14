import numpy as np
import faiss
import os
import time

# Function to load numpy arrays and concatenate them
def load_and_concat_arrays(folder_path, num_files):
    arrays = []
    for i in range(num_files):
        file_path = os.path.join(folder_path, f'{i:05d}.npy')
        array = np.load(file_path)
        arrays.append(array)
    concatenated_array = np.vstack(arrays)
    # normalize that
    concatenated_array = concatenated_array / np.linalg.norm(concatenated_array, axis=1)[:, None]
    return concatenated_array

t0 = time.time()

# Load and concatenate arrays
folder_path = '../sscdemb'
num_files = 11357
data = load_and_concat_arrays(folder_path, num_files)
print(f"Shape of concatenated data: {data.shape}, time taken: {time.time() - t0:.2f}s")

# Number of clusters
num_clusters = 16000

# Initialize Faiss k-means
d = data.shape[1]
kmeans = faiss.Kmeans(d, num_clusters, niter=10, verbose=True, max_points_per_centroid = 50)

# Perform k-means clustering
kmeans.train(data)

# Get cluster assignments and centroids
D, I = kmeans.index.search(data, 1)  # D is the distance array, I is the index array
centroids = kmeans.centroids

# Create output directory if it does not exist
output_dir = '../sscd_cluster_info'
os.makedirs(output_dir, exist_ok=True)

# Save cluster indices and centroids
for i in range(num_clusters):
    cluster_indices = np.where(I == i)[0]
    cluster_indices_path = os.path.join(output_dir, f'cidx_{i}.npy')
    np.save(cluster_indices_path, cluster_indices)
    
    centroid_path = os.path.join(output_dir, f'cemb_{i}.npy')
    # save the cluster's embedding for all indices in the cluster
    cluster_emb = data[cluster_indices]
    np.save(centroid_path, cluster_emb)