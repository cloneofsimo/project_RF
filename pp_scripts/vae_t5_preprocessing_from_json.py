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


def prepare_image(pil_image, w):
    thisw, thish = pil_image.size
   
    # resize smaller side to w
    if thisw < thish:
        pil_image = pil_image.resize((w, int(w * thish / thisw)), resample=Image.BICUBIC)
    else:
        pil_image = pil_image.resize((int(w * thisw / thish), w), resample=Image.BICUBIC)

    pil_image = crop_to_center(pil_image, w)
    # pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image

def image_interestingness(image: Image.Image) -> float:
    image = image.convert('L')
    image_np = np.array(image)
    f_transform = np.fft.fft2(image_np)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)
    normalized_spectrum = log_magnitude_spectrum / np.sum(log_magnitude_spectrum)
    entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + np.finfo(float).eps))
    return entropy



class JsonDataset(Dataset):
    def __init__(self, json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = list(json.load(f).items())
        self.json_file_path = json_file_path
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        key, value = self.data[idx]
        filename = value['filename']
        imgpath = os.path.join(os.path.dirname(self.json_file_path), filename)

        caption = value['cogvlm_caption']
        try:
            image = Image.open(imgpath).convert("RGB")
            interestingness = image_interestingness(image)
        except:
            image = Image.new("RGB", (256, 256), (255, 255, 255))
            interestingness = 0.0
        
        h, w = image.size
        if h < 1024 or w < 1024:
            interestingness = 0.0
        
        image = prepare_image(image, 1024)
        
        return {
            "image": image,
            "caption": caption,
            "interestingness": interestingness
        }
    



from tqdm import tqdm
from torch.utils.data import DataLoader
import webdataset as wds
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM




@torch.no_grad()
def convert_to_mds(
    dataset_paths, out_roots, device, is_test=False, selective_jsons=None
):
    logging.info(f"Processing on {device}")
    # Load the VAE model
    vae_model = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    vae_model = vae_model.to(device).eval()
    vae_model.to(memory_format=torch.channels_last)
    
    t5tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-xl", use_fast=False)
    t5tokenizer.pad_token = t5tokenizer.bos_token
    t5model = AutoModelForSeq2SeqLM.from_pretrained("EleutherAI/pile-t5-xl")
    t5model = t5model.to(device).eval()

    for dataset_path, out_root, selective_json in zip(dataset_paths, out_roots, selective_jsons):
        if not os.path.exists(dataset_path):
            logging.info(f"Dataset not found: {dataset_path}")
            return

        dataset = JsonDataset(dataset_path)

        dataloader = DataLoader(dataset, batch_size=32, num_workers=4, prefetch_factor = 3)

        if selective_json is not None:
            with open(selective_json) as f:
                selective_json = json.load(f)



        t0 = time.time()
        sub_data_root = os.path.join(out_root, "data")
        columns = {"vae_1024x1024_latents": "np16", "caption": "str", 't5_xl_embeddings': 'uint8'}

        if os.path.exists(sub_data_root):
            # Remove all files in the directory
            for file in os.listdir(sub_data_root):
                os.remove(os.path.join(sub_data_root, file))
        os.makedirs(sub_data_root, exist_ok=True)

        with MDSWriter(out=sub_data_root, columns=columns) as out:
            inference_latencies = []

            for idx, batch in tqdm(enumerate(dataloader)):
                batchidx = range(idx * 32, (idx + 1) * 32)
                
              

                start_time = time.time()

                processed_images, captions, interestingness = batch["image"], batch["caption"], batch["interestingness"]
                if selective_json is not None:
                    batch_sel_idx = [x - idx * 32 for x in batchidx if (x in selective_json)]
                    batch_sel_idx = torch.tensor(batch_sel_idx)
                    if len(batch_sel_idx) == 0:
                        continue
                else:
                    batch_sel_idx = torch.arange(len(captions))
                    # interestingness > 5.0
                    batch_sel_idx = batch_sel_idx[interestingness > 5.0]

                # select 
                processed_images = processed_images[batch_sel_idx]
                captions = [captions[i] for i in batch_sel_idx]
                
                ### VAE
                image_for_vae = processed_images.to(device).half()
                vae_latents = vae_model.encode(image_for_vae).latent_dist.sample()
                vae_outputs = vae_latents.cpu().numpy().astype(np.float16)


                # T5
                t5_inputs = t5tokenizer(
                    captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                t5_inputs = {k: v.to(device) for k, v in t5_inputs.items()}
                t5_outputs = t5model.encoder(**t5_inputs)[0]
                mask = (
                    t5_inputs["attention_mask"].unsqueeze(-1).expand(t5_outputs.shape)
                )
                t5_outputs = t5_outputs * mask
                t5_outputs = (
                    ((t5_outputs.clip(-0.25, 0.25) / 0.5 + 0.5) * 255.0)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                # Iterate through the batch
                for i in range(len(captions)):
                    sample = {
                        "vae_1024x1024_latents": vae_outputs[i],
                        "caption": str(captions[i]),
                        "t5_xl_embeddings": t5_outputs[i],
                    }
                    out.write(sample)

                inference_latencies.append(time.time() - start_time)

                if is_test:
                    break

            logging.info(
                f"Average Inference Latency on {device}: {np.mean(inference_latencies)} seconds"
            )
            logging.info(
                f"Total Inference Time on {device}: {time.time() - t0} seconds"
            )


def main(
    datasetinfos,
    out_roots,
    is_test=False,
    device_name="cuda",
    selective_jsons=None
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Processing on {device}")
    convert_to_mds(
        datasetinfos, out_roots, device, is_test=is_test, selective_jsons = selective_jsons
    )
    logging.info("Finished processing images.")


import argparse

import os
import json

def detect_small_or_nonexistent_dirs(current_dir, start=0, end=14000, max_size=1024):
    small_or_nonexistent_dirs = []

    
    for i in range(start, end + 1):
        dir_name = f"{i:05d}"
        dir_path = os.path.join(current_dir, dir_name)
        
        if not os.path.exists(dir_path):
            small_or_nonexistent_dirs.append(i)
        elif os.path.isdir(dir_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            if total_size < max_size:
                small_or_nonexistent_dirs.append(i)
    
    return small_or_nonexistent_dirs

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


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
    import json
    # small_dirs = detect_small_or_nonexistent_dirs()
    # save_to_json(small_dirs, 'small_or_nonexistent_dirs.json')

    #reqsids = json.load(open("/home/host/simo/capfusion_mds/small_or_nonexistent_dirs.json"))

    out_roots, datasetinfos, selective_jsons = [], [], []
    for i, reqid in enumerate(range(1, 11)):
        if i % 8 == args.file_index:
            out_root = f"../laionmds_t5xl/{str(int(reqid)).zfill(5)}"
            dataset_path = f"/home/host/simo/laionpop/images/chunk_{int(reqid)}/chunk_{int(reqid)}.0.json"
            selective_json = None
            out_roots.append(out_root)
            datasetinfos.append(dataset_path)
            selective_jsons.append(selective_json)
            
            
    main(
        datasetinfos,
        out_roots,
        is_test=args.is_test,
        device_name=args.device,
        selective_jsons=selective_jsons
    )
