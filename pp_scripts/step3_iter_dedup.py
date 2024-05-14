import numpy as np
import faiss
import os
from autofaiss import build_index
import tempfile
import unittest

import torch


def chunked_cdist(vecs, chunk_size, device):
    n = vecs.size(0)
    dist = torch.zeros((n, n), dtype=vecs.dtype, device='cpu')  # Initialize on CPU
    
    # Iterate over chunks
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            # Move the chunks to the GPU
            vecs_chunk_1 = vecs[i:i+chunk_size].to(device)
            vecs_chunk_2 = vecs[j:j+chunk_size].to(device)
            
            # Compute pairwise distances between the chunks
            dist_chunk = torch.cdist(vecs_chunk_1, vecs_chunk_2, p=2)
            
            # Move the result back to CPU and store it in the appropriate slice
            dist[i:i+chunk_size, j:j+chunk_size] = dist_chunk.cpu()
    
    return dist

def dedup_cluster(cluster_vec_emb_file, cluster_item_file, i, sim_thres = 0.4):

    vecs = np.load(cluster_vec_emb_file)
    # normalize vector B, D
    vecs = vecs / np.linalg.norm(vecs, axis=1)[:, None]

    #print(vecs)
    items = np.load(cluster_item_file)

    #print(items)
    
    dups = set()
    online_mostduped_set = []
    mostduped_set_len = -1

    if len(vecs) < 30000: # its gona need 1.6G of GPU memory, im good with that.
        # make torch cuda tensor of all the pairwise distance between the vectors, in l2
        # then use torch to find the indices of the pairs that are less than sim_thres
        # then add the indices to the dups set
        # then remove the dups from the items
        # then sort the items
        # then return the items
        
        vecs = torch.tensor(vecs).to(f'cuda:{i%8}').float()
        #dist = torch.cdist(vecs, vecs, p=2)
        dist = chunked_cdist(vecs, 2000, f'cuda:{i%8}')
        dups = set()
        for i, item in enumerate(items):
            if item in dups:
                continue
            dup_indices = torch.where(dist[i] < sim_thres)[0]
            dup_ids = set()
            for j in dup_indices:
                if items[j] != item:
                    dup_ids.add(items[j])
            if len(dup_ids) > mostduped_set_len:
                mostduped_set_len = len(dup_ids)
                online_mostduped_set = dup_ids | set([item])
            dups.update(dup_ids)

    else:
        index, index_infos = build_index(vecs, save_on_disk=False, metric_type='l2')
        for i, (vec, item) in enumerate(zip(vecs, items)):
            if item in dups:
                continue

            lims, D, I = index.range_search(vec.reshape(1, -1), sim_thres)

            start, end = lims[0], lims[1]
            dup_indices = I[start:end]
            dup_ids = set()
            for j in dup_indices:
                if items[j] != item:
                    dup_ids.add(items[j])

            if len(dup_ids) > mostduped_set_len:
                mostduped_set_len = len(dup_ids)
                online_mostduped_set = dup_ids | set([item])
            
            dups.update(dup_ids)

    deduped_items = set(items) - dups
    deduped_items = list(deduped_items)
    deduped_items.sort()

    return deduped_items, online_mostduped_set

class TestDedupCluster(unittest.TestCase):

    def test_dedup_cluster(self):
        # Create some sample data
        vecs = np.array([[1, 2, 3], [1.1, 2.1, 3.1], [1.1, 2.1, 3.3], [1.1, 2.1, 5.0], [-4, 5, 6],  [-4, 5, 20]])
        # normalize
        vecs = vecs / np.linalg.norm(vecs, axis=1)[:, None]
        items = np.array(["a", "b", "bx", "bxx", "c", "d"])

        # Save the data to temporary files
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as vec_file:
            np.save(vec_file, vecs)
            vec_file_name = vec_file.name
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as item_file:
            np.save(item_file, items)
            item_file_name = item_file.name

        # Dedup the cluster
        deduped_items, online_mostduped = dedup_cluster(vec_file_name, item_file_name, sim_thres=0.95)

        # Check the results
        print(deduped_items)
        print(online_mostduped)
        self.assertTrue(np.array_equal(deduped_items, np.array(["a", "c", "d"])))

        # Clean up the temporary files
        os.unlink(vec_file_name)
        os.unlink(item_file_name)


import json

def run(i):
    print(f"Doing {i} cluster")
    cluster_vec_emb_file = f'../sscd_cluster_info/cemb_{i}.npy'
    cluster_item_file = f'../sscd_cluster_info/cidx_{i}.npy'
    len_c = len(np.load(cluster_item_file))
    # check if it is already done
    if os.path.exists(f'../sscd_dedup_cluster_info/mostduped_{i}.json'):
        print(f"Cluster {i} already deduped")
        return

    deduped_items, mostdeduped = dedup_cluster(cluster_vec_emb_file, cluster_item_file, i)
    # mostdeduped to just json
    with open(f'../sscd_dedup_cluster_info/mostduped_{i}.json', 'w') as f:
        mostdeduped = [int(x) for x in mostdeduped]
        json.dump(list(mostdeduped), f)
    np.save(f'../sscd_dedup_cluster_info/cidx_{i}.npy', deduped_items)
    
    print(f"Reduced the cluster {i} from {len_c} to {len(deduped_items)}, {100 * (len_c - len(deduped_items)) / len_c}%")



if __name__ == "__main__":
    #unittest.main()
    # import json
    # # launch multiple processes, use pbar to track process.
    count_cluster = 16000
    count_tar = 11357
    import multiprocessing
    from tqdm import tqdm
    import numpy as np
    import os
    import json
    from collections import deque
    with multiprocessing.Pool(8) as p:
        list(tqdm(p.imap(run, range(count_cluster)), total=count_cluster))
    import time
    t0 = time.time()
    
    print(f"Time taken: {time.time() - t0} seconds")

    # now gather all the embedding info, and sort them
    all_deduped_items = []
    for i in range(count_cluster):
        deduped_items = np.load(f'../sscd_dedup_cluster_info/cidx_{i}.npy')
        all_deduped_items.extend(deduped_items.tolist())

    # emb size per tar
    emb_sizes = []
    for tidx in range(count_tar):
        emb_size = len(np.load(f'../sscdemb/{tidx:05d}.npy'))
        print(f"Tar {tidx} emb size: {emb_size}")
        emb_sizes.append(emb_size)

    # Sort the all_deduped_items, and save them per offset
    os.makedirs('../sscd_deduped_pertar_items', exist_ok=True)
    all_deduped_items.sort()
    all_deduped_items = deque(all_deduped_items)  # Use deque for efficient popping from the front

    offset = 0
    for tidx, emb_size in enumerate(emb_sizes):
        offset += emb_size

        to_save_items = []
        while len(all_deduped_items) > 0 and all_deduped_items[0] < offset:
            to_save_items.append(all_deduped_items.popleft() - offset + emb_size)  # Use popleft for efficient removal

        with open(f'../sscd_deduped_pertar_items/{tidx:05d}.json', 'w') as f:
            json.dump(to_save_items, f)