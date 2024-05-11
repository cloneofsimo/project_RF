# based on preprocessed sscd, check the most similar input to the monarisa image.

import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from diffusers.models import AutoencoderKL
from streaming import MDSWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
import webdataset as wds
from streaming import StreamingDataset
from torchvision import transforms

import logging
import time
import numpy as np
from typing import Any

def crop_to_center(image, new_size=768):
    width, height = image.size

    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def totorch(img):
    arr = np.array(img.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image


def prepare_image(pil_image, w=512, h=512):
    thisw, thish = pil_image.size
    #assert thisw == 256 or thish == 256, f"Image size is {thisw}x{thish}"
    pil_image = crop_to_center(pil_image, 256)

    # make colorbin histogram
    twoxtwo = pil_image.resize((2, 2), Image.NEAREST)
    fourxfour = pil_image.resize((4, 4), Image.NEAREST)

    image = small_288(pil_image)
    return image, totorch(twoxtwo), totorch(fourxfour)


# Initialize logging
logging.basicConfig(level=logging.INFO)


from streaming.base.format.mds.encodings import Encoding, _encodings


class np16(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.float16)


_encodings["np16"] = np16

device = "cuda:0"

model = torch.jit.load("sscd_disc_mixup.torchscript.pt").to(device)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose(
    [
        transforms.Resize(288),
        transforms.ToTensor(),
        normalize,
    ]
)

# load image
img = Image.open("./histogram.png").convert("RGB")
processed_images, twoxtwo, fourxfour = prepare_image(img)
vec = model(processed_images.to(device).unsqueeze(0)).cpu().detach().numpy()

local_train_dir = "/home/host/simo/sscd"

train_dataset = StreamingDataset(
    local=local_train_dir,
    remote=None,
    split=None,
    shuffle=True,
    shuffle_algo="naive",
    num_canonical_nodes=1,
    batch_size=32,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=3,
)


first_batch = batch = next(iter(train_dataloader))
print(first_batch['sscd_vector_output'].shape)

cossims = []
captions = []

for dl in train_dataloader:
    vecs = dl['sscd_vector_output']
    # vec is 1, 512
    # vecs is 32, 512

    cossim = np.dot(vec, vecs.T) / (np.linalg.norm(vec) * np.linalg.norm(vecs, axis=1))
    
    cossims.extend(cossim.squeeze().tolist())
    captions.extend(dl['caption'])

    if len(cossims) > 100000:
        break

# sort by cossim
cossims = np.array(cossims)
captions = np.array(captions)
idx = np.argsort(cossims)[::-1]
cossims = cossims[idx]
captions = captions[idx]

print(captions[0], cossims[0])
