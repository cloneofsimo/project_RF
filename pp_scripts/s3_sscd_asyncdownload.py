import os
import boto3
import time
import multiprocessing
from multiprocessing import Pool

def download_from_s3(bucket, key, download_path):
    s3 = boto3.client(
        's3',
        endpoint_url='https://fly.storage.tigris.dev',
        region_name='auto',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    s3.download_file(bucket, key, download_path)
    print(f"Downloaded {key} to {download_path}")

def download_and_check(args):
    dataset_path, local_dir = args
    if dataset_path.startswith("s3"):
        s3_removed = dataset_path.replace("s3://", "")
        bucket, key = s3_removed.split("/", 1)
        download_path = os.path.join(local_dir, os.path.basename(key))
        download_from_s3(bucket, key, download_path)
        # Create a notification file to signal that the download is complete
        with open(f"{download_path}.done", "w") as f:
            f.write("done")
        print(f"Download complete for {download_path}")

        # check the storage usage. if its more than 50%, wait for 1 min
        while True:
            storage = os.statvfs(local_dir)
            if storage.f_bavail / storage.f_blocks > 0.5:
                print("Storage usage is less than 50%. Continuing...")
                break
            print("Storage usage is more than 50%. Waiting for 1 min...")
            time.sleep(60)

def main(dataset_paths, local_dir):
    os.makedirs(local_dir, exist_ok=True)

    # Using multiprocessing to download files concurrently
    num_workers = 16
    with Pool(num_workers) as p:
        p.map(download_and_check, [(dataset_path, local_dir) for dataset_path in dataset_paths])

if __name__ == "__main__":
    root_dir = os.environ.get("ROOT_DIR", "s3://datacomp-1b/datacomp-wds/datacomp-wds")
    local_dir = "../temp"
    dataset_paths = [f"s3://{root_dir}/{str(idx).zfill(5)}.tar" for idx in range(30, 20001)]

    main(dataset_paths, local_dir)
