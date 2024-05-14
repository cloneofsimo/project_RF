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


def prepare_image(pil_image, w=512, h=512):
    thisw, thish = pil_image.size
    assert thisw == 256 or thish == 256, f"Image size is {thisw}x{thish}"
    pil_image = crop_to_center(pil_image, 256)
    # pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image


def preprocess(x):
    image, caption = x
    print(image, caption)
    image = prepare_image(image, 256, 256)
    # # Assuming the caption is in a JSON field
    caption = caption["caption"]
    return {"image": image, "caption": caption}


from tqdm import tqdm
from torch.utils.data import DataLoader
import webdataset as wds
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM




@torch.no_grad()
def convert_to_mds(
    dataset_path, out_root, device, is_test=False, selective_json=None
):
    logging.info(f"Processing on {device}")
    # Load the VAE model
    vae_model = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    vae_model = vae_model.to(device).eval()
    vae_model.to(memory_format=torch.channels_last)
    
    t5tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-large", use_fast=False)
    t5tokenizer.pad_token = t5tokenizer.bos_token
    t5model = AutoModelForSeq2SeqLM.from_pretrained("EleutherAI/pile-t5-large")
    t5model = t5model.to(device).eval()

    # vae_model.encode = torch.compile(vae_model.encode, mode="reduce-overhead", fullgraph=False)
    dataset = wds.WebDataset(dataset_path).decode("pil").to_tuple("jpg;png", "json")

    # Create the dataset and dataloader
    dataset = dataset.map(preprocess)

    # if dataset.__len__() < 1:
    #     logging.info("No images to process.")
    #     return

    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

    if selective_json is not None:
        with open(selective_json) as f:
            selective_json = json.load(f)




    sub_data_root = os.path.join(out_root, "data")
    columns = {"vae_output": "uint8", "caption": "str", 't5emb': 'np16'}

    if os.path.exists(sub_data_root):
        # Remove all files in the directory
        for file in os.listdir(sub_data_root):
            os.remove(os.path.join(sub_data_root, file))
    os.makedirs(sub_data_root, exist_ok=True)

    with MDSWriter(out=sub_data_root, columns=columns) as out:
        inference_latencies = []

        for idx, batch in tqdm(enumerate(dataloader)):
            batchidx = range(idx * 32, (idx + 1) * 32)
            
            if selective_json is not None:
                batch_sel_idx = [x - idx * 32 for x in batchidx if (x in selective_json)]
                batch_sel_idx = torch.tensor(batch_sel_idx)
                if len(batch_sel_idx) == 0:
                    continue

            start_time = time.time()

            processed_images, captions = batch["image"], batch["caption"]
            # select 
            processed_images = processed_images[batch_sel_idx]
            captions = [captions[i] for i in batch_sel_idx]
            
            # VAE
            processed_images = processed_images.to(device).half()
            vae_outputs = vae_model.encode(processed_images).latent_dist.sample()

            vae_outputs = (vae_outputs.clip(-14, 14) / 28.0 + 0.5) * 255.0
            vae_outputs = vae_outputs.to(torch.uint8)

            # T5
            t5_inputs = t5tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=128)
            t5_inputs = {k: v.to(device) for k, v in t5_inputs.items()}
            t5_outputs = t5model.encoder(**t5_inputs)[0] # B, T, D
            # mask that by 0 for padding tokens
            mask = t5_inputs["attention_mask"].unsqueeze(-1).expand(t5_outputs.shape)
            t5_outputs = t5_outputs * mask

            # Iterate through the batch
            for i in range(len(captions)):
                sample = {
                    "vae_output": vae_outputs[i].cpu().numpy().astype(np.uint8),
                    "caption": str(captions[i]),
                    't5emb': t5_outputs[i].cpu().numpy().astype(np.float16)
                }
                out.write(sample)

            inference_latencies.append(time.time() - start_time)

            if is_test:
                break

        logging.info(
            f"Average Inference Latency on {device}: {np.mean(inference_latencies)} seconds"
        )


def main(
    datasetinfo,
    out_root,
    is_test=False,
    device_name="cuda",
    selective_json=None
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Processing on {device}")
    convert_to_mds(
        datasetinfo, out_root, device, is_test=is_test, selective_json = selective_json
    )
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
        "--file_index", type=int, default=1, help="File index to process."
    )
    parser.add_argument(
        "--is_test", action="store_true", help="Run in test mode with reduced dataset."
    )

    args = parser.parse_args()

    out_root = f"/home/host/simo/capfusion_mds/{str(args.file_index).zfill(5)}"
    dataset_path = f"/home/host/simo/capfusion_256/{str(args.file_index).zfill(5)}.tar"
    selective_json = f"/home/host/simo/sscd_deduped_pertar_items/{str(args.file_index).zfill(5)}.json"
    
    main(
        dataset_path,
        out_root,
        is_test=args.is_test,
        device_name=args.device,
        selective_json=selective_json
    )
