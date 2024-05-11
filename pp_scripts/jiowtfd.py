import json
import os
from PIL import Image
import webdataset as wds

class DatasetLocator:
    def __init__(self, dataset_path_pattern, json_pattern):
        self.dataset_path_pattern = dataset_path_pattern
        self.json_pattern = json_pattern
        self.index_map = self._build_index_map()

    def _build_index_map(self):

        cumulative_size_list = []
        cursize = 0
        for i in range(1000):
            json_path = self.json_pattern.format(i)
            with open(json_path, 'r') as f:
                data = json.load(f)
                
                size = data['successes']
                cursize += size
                cumulative_size_list.append((i, size, cursize))

        return cumulative_size_list

    def __getitem__(self, idx):
        
        # first find where does idxth data belong?
        for i, size, cumsize in self.index_map:
            if idx < cumsize:
                return self._load_item(i, idx - (cumsize - size))


    def _load_item(self, tar_idx, offset):
        tar_path = self.dataset_path_pattern.format(tar_idx)
        dataset = wds.WebDataset(tar_path).decode("pil").to_tuple("jpg;png", "json")
        for idx, (image, caption) in enumerate(dataset):
            if idx == offset:
                return image, caption
            

# Usage
locator = DatasetLocator(
    dataset_path_pattern="/home/host/simo/capfusion_256/{:05d}.tar",
    json_pattern="/home/host/simo/capfusion_256/{:05d}_stats.json"
)


candidates = set([1332656, 1341506, 104645, 1251693, 1321899, 476365, 1323967])

for idx in candidates:
    image, caption = locator[idx]
    print(idx)
    print(image)
    print(caption)
    image.save(f"./temp/image_{idx}.png")