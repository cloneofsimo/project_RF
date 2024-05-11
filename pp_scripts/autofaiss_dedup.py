import faiss
import glob
import numpy as np


def find_duplicates(vecs, items, index_path, thresh = 0.3):
    # Load the FAISS index
    my_index = faiss.read_index(glob.glob(index_path)[0])

    dups = set()  # Store duplicate item IDs

    for i, vec in enumerate(vecs):
        qs = np.float32(vec).reshape(1, -1)  # Convert vector to float32
        qid = items[i]  # Current item ID
        
        # Perform the search
        lims, D, I = my_index.range_search(qs, thresh)

        if qid in dups:
            continue

        # Collect all indices within the threshold range
        duplicate_indices = I[lims[0]:lims[1]]
        duplicate_ids = []

        for j in duplicate_indices:
            if items[j] != qid:
                duplicate_ids.append(items[j])

        dups.update(duplicate_ids)

    return dups

vecs = np.load('../sscdemb/00000.npy')
print(vecs.shape)
dups = find_duplicates(vecs, list(range(len(vecs))), index_path = "../sscd_index/*.index" )
print(dups)