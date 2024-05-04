import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from diffusers.models import AutoencoderKL
from streaming import MDSWriter

import logging
import time
import numpy as np
from typing import Any

# Initialize logging
logging.basicConfig(level=logging.INFO)


from streaming.base.format.mds.encodings import Encoding, _encodings


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.uint8)


class np16(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.float16)


_encodings["np16"] = np16
_encodings["uint8"] = uint8


def crop_to_center(image, new_size=768):
    width, height = image.size

    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


from tqdm import tqdm
from torch.utils.data import DataLoader
import webdataset as wds

from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])
def totorch(img):
    arr = np.array(img.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image

def prepare_image(pil_image, w=512, h=512):
    thisw, thish = pil_image.size
    assert (thisw == 256 or thish == 256), f"Image size is {thisw}x{thish}"
    pil_image = crop_to_center(pil_image, 256)

    # make colorbin histogram
    twoxtwo = pil_image.resize((2, 2), Image.NEAREST)
    fourxfour = pil_image.resize((4, 4), Image.NEAREST)

    image = small_288(pil_image)
    return image, totorch(twoxtwo), totorch(fourxfour)


def preprocess(x):
    image, caption = x
    #print(image, caption)
    image, twoxtwo, fourxfour  = prepare_image(image, 256, 256)
    # # Assuming the caption is in a JSON field
    caption = caption["caption"]
    return {"image": image, "caption": caption, "twoxtwo": twoxtwo, "fourxfour": fourxfour}



@torch.no_grad()
def convert_to_mds(
    dataset_path, out_root, device, batch_size=8, num_workers=4, is_test=False
):
    logging.info(f"Processing on {device}")

    model = torch.jit.load("sscd_disc_mixup.torchscript.pt").to(device)
    #vae_model.encode = torch.compile(vae_model.encode, mode="reduce-overhead", fullgraph=False)
    dataset = wds.WebDataset(dataset_path).decode("pil").to_tuple("jpg;png", "json")

    # Create the dataset and dataloader
    dataset = dataset.map(preprocess)
    
    # if dataset.__len__() < 1:
    #     logging.info("No images to process.")
    #     return

    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

    sub_data_root = os.path.join(out_root, "data")
    columns = {"sscd_vector_output": "np16", "caption": "str", 'twoxtwo': 'np16', 'fourxfour': 'np16'}

    if os.path.exists(sub_data_root):
        # Remove all files in the directory
        for file in os.listdir(sub_data_root):
            os.remove(os.path.join(sub_data_root, file))
    os.makedirs(sub_data_root, exist_ok=True)

    with MDSWriter(out=sub_data_root, columns=columns) as out:
        inference_latencies = []

        for batch in tqdm(dataloader):
            start_time = time.time()

            processed_images, captions = batch["image"], batch["caption"]
            twoxtwo, fourxfour = batch['twoxtwo'], batch['fourxfour']
            

            embeddings = model(processed_images.to(device))
            
            for i in range(len(captions)):
                sample = {
                    "sscd_vector_output": embeddings[i].cpu().numpy().astype(np.float16),
                    "caption": str(captions[i]),
                    'twoxtwo': twoxtwo[i].cpu().numpy().astype(np.float16),
                    'fourxfour': fourxfour[i].cpu().numpy().astype(np.float16),
                }
                out.write(sample)

            inference_latencies.append(time.time() - start_time)

            if is_test:
                break

        logging.info(
            f"Average Inference Latency on {device}: {np.mean(inference_latencies)} seconds"
        )


def main(
    datasetinfo, out_root, batch_size=64, num_workers=8, is_test=False, device_name="cuda"
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Processing on {device}")
    convert_to_mds(datasetinfo, out_root, device, batch_size, num_workers, is_test=is_test)
    logging.info("Finished processing images.")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to MDS format.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for processing (cuda or cpu).",
    )
    parser.add_argument(
        "--file_index", type=int, default=0, help="File index to process."
    )
    parser.add_argument(
        "--is_test", action="store_true", help="Run in test mode with reduced dataset."
    )

    args = parser.parse_args()

    out_root = f"/home/host/simo/sscd/{str(args.file_index).zfill(5)}"
    dataset_path=f"/home/host/simo/capfusion_256/{str(args.file_index).zfill(5)}.tar"
    #out_root = "./here"
    main(
        dataset_path,
        out_root,
        is_test=args.is_test,
        device_name=args.device,
    )