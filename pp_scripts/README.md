# Preprocessing: Download, preprocess to VAE, vectorized with SSCD, dedup using FAISS

Before starting, come to this dir.

```bash
cd pp_scripts
```

## Step 1.

First, use webdataset to download a really large whatever .tar lists.

* Example: I've donwloaded capfusion dataset. Running `bash run_img2dataset_capfusion.sh` will download from url of the parquets. Output file is `../capfusion_256`.

## Step 2 make SSCD embedding of the images.

Make SSCD embeddings of the images. This shouldn't take long for 50M images.

`step2_run_sscd.sh` will automatically do that, but you have to specifiy output and input dir.

## Step 3 Cluster & Dedup using FAISS

Now install `pip install autofaiss`, come to this directory, and run

`python step3_make_cluster.py`

to make 16,000 clusters.

Then run `python step3_iter_dedup.py` to deduplicate based on the clusters. This will save deduplicated image ids in ../sscd_dedup_cluster_info, as well as "maximum duplicates" as a sanity check.



