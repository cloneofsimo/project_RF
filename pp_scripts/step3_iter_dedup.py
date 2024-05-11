import numpy as np
import faiss
import os
from autofaiss import build_index
import tempfile
import unittest

def dedup_cluster(cluster_vec_emb_file, cluster_item_file, sim_thres = 0.1):
    vecs = np.load(cluster_vec_emb_file)
    # normalize vector B, D
    vecs = vecs / np.linalg.norm(vecs, axis=1)[:, None]

    #print(vecs)
    items = np.load(cluster_item_file)

    #print(items)

    dups = set()
    online_mostduped_set = []
    mostduped_set_len = -1
    index, index_infos = build_index(vecs, save_on_disk=False, metric_type='l2')
    for i, (vec, item) in enumerate(zip(vecs, items)):
        if item in dups:
            continue
        lims, D, I = index.range_search(vec.reshape(1, -1), sim_thres)
       

        #print(D, I)
        #print(D)

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


if __name__ == "__main__":
    #unittest.main()
    import json



    for i in range(400):
        print(f"Doing {i} cluster")
        cluster_vec_emb_file = f'../sscd_cluster_info/cemb_{i}.npy'
        cluster_item_file = f'../sscd_cluster_info/cidx_{i}.npy'
        deduped_items, mostdeduped = dedup_cluster(cluster_vec_emb_file, cluster_item_file)
        # mostdeduped to just json
        with open(f'../sscd_dedup_cluster_info/mostduped_{i}.json', 'w') as f:
            mostdeduped = [int(x) for x in mostdeduped]
            json.dump(list(mostdeduped), f)
        np.save(f'../sscd_dedup_cluster_info/cidx_{i}.npy', deduped_items)

        
