from streaming import StreamingDataset
import os
import shutil
import torch

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor


remote_train_dir = None
local_train_dir = "/home/host/simo/laionmds_t5xl"
from streaming.base.format.mds.encodings import Encoding, _encodings
import numpy as np

class np16(Encoding):
    def encode(self, obj) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes):
        return np.frombuffer(data, np.float16)


class uint8(Encoding):
    def encode(self, obj) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes):
        return np.frombuffer(data, np.uint8)


_encodings["uint8"] = uint8


_encodings["np16"] = np16

# if os.path.exists(local_train_dir):
#     shutil.rmtree(local_train_dir)

train_dataset = StreamingDataset(
    local=local_train_dir,
    remote=remote_train_dir,
    split=None,
    shuffle=True,
    shuffle_algo="naive",
    num_canonical_nodes=1,
    batch_size=32,
)

COLUMNS = {
    "caption": "str",
    "vae_1024x1024_latents": "np16",
    "t5_xl_embeddings": "uint8",
}

# except for pil, everything is in the batch
def collate_fn(batch):
    k = list(batch[0].keys())
    out = {}
    for key in k:
        if key == "image":
            out[key] = [b[key] for b in batch]
        else:
            out[key] = [b[key] for b in batch]
        # try torch 
        try:
            out[key] = torch.tensor(out[key])
        except:
            pass
    return out
    

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=3,
    collate_fn=collate_fn,
)



batch = next(iter(train_dataloader))

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to("cuda:0")


i = 10
vae_latent = batch["vae_1024x1024_latents"].reshape(-1, 4, 32 * 4, 32 * 4)[i : i + 1, :, 16:64, 16:64].cuda().float()
# normalize so that its in -127, 127. min-max via min : -12, max : 12
# vae_latent = (vae_latent.clip(-12, 12) / 24.0 + 0.5) * 255.0
# # as int 8
# vae_latent = vae_latent.to(torch.uint8)
# print(vae_latent)

# # ok now reverse
# vae_latent = vae_latent.to(torch.float32)
#vae_latent = (vae_latent / 255.0 - 0.5) * 24.0
# vae_latent = vae_latent.cuda()

# check out average size of the latent
print(vae_latent.mean(), vae_latent.std(), vae_latent.min(), vae_latent.max())
# print the quantiles
for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
    print(f"Quantile {q}: {torch.quantile(vae_latent, q)}")

x = vae.decode(vae_latent.cuda()).sample
img = VaeImageProcessor().postprocess(image=x.detach(), do_denormalize=[True, True])[0]
caption = batch["caption"][i]

img.save("test.png")
print(caption)

# debug     "t5_xl_embeddings": "uint8",
    # "sscd_embeddings": "np16",
    # "hps_score": "str",

t5emb = torch.tensor(batch["t5_xl_embeddings"][i]).float()
print(t5emb.dtype, t5emb.shape, t5emb.min(), t5emb.max())
for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
    print(f"Quantile {q}: {torch.quantile(t5emb, q)}")
