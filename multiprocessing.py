#!/usr/bin/env python
# coding: utf-8



# 🔥 CELL 1 — Setup paths

import json, random, shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT = Path('/Users/epsilon_ai/DINO_CNX_ImageNet')

TRAIN_ROOT = ROOT / 'ImageNet/ILSVRC2012_img_train'
SMOKE_ROOT = ROOT / 'SmokeTest'
SMOKE_SPLIT = SMOKE_ROOT / 'splits'
SMOKE_SHARDS = SMOKE_ROOT / 'shards'

SMOKE_SPLIT.mkdir(parents=True, exist_ok=True)
SMOKE_SHARDS.mkdir(parents=True, exist_ok=True)

SEED = 999
random.seed(SEED)

print("Smoke-test folders ready.")


# In[8]:


from pathlib import Path
import random

ROOT = Path("/Users/epsilon_ai/DINO_CNX_ImageNet")
train_root = ROOT / "ImageNet/ILSVRC2012_img_train"
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}

random.seed(123)

def list_class_images(cls_dir):
    return sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

# collect 5 images from each class
smoke_paths = []

for wnid_dir in sorted(train_root.iterdir()):
    if not wnid_dir.is_dir():
        continue
    
    files = list_class_images(wnid_dir)
    if len(files) == 0:
        continue
    
    chosen = random.sample(files, k=min(5, len(files)))   # choose 5 per class
    smoke_paths.extend(chosen)

print("SMOKE dataset size:", len(smoke_paths))
print("Example:", smoke_paths[0])


# In[10]:


import os
import multiprocessing as mp
from PIL import Image
from io import BytesIO

# --- make sure macOS uses 'spawn' ---
mp.set_start_method("spawn", force=True)

# --- define worker function at top-level ---
def _read_one(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(BytesIO(f.read())).convert("RGB")
        return path, img
    except Exception as e:
        return path, None

def load_images_multiproc(paths, num_workers=8):
    with mp.Pool(processes=num_workers) as pool:
        out = list(pool.imap_unordered(_read_one, paths))
    return out


# --- Smoke-test ---
if __name__ == "__main__":
    print("Loading SMOKE subset:", len(smoke_paths), "images")
    smoke_samples = load_images_multiproc(smoke_paths, num_workers=12)

    ok = sum(1 for _,img in smoke_samples if img is not None)
    bad = len(smoke_samples) - ok

    print("Loaded:", ok, "OK,", bad, "failed")


# In[9]:


import os
import multiprocessing as mp

def _read_one(path):
    try:
        with open(path, "rb") as f:
            return (os.path.basename(path), f.read())
    except:
        return None

def load_images_multiproc(paths, num_workers=22):
    with mp.Pool(num_workers) as pool:
        out = pool.map(_read_one, paths)
    # Drop None entries
    out = [ {"filename": fn, "bytes": data} 
            for fn, data in out if fn is not None ]
    return out

# --- RUN ON SMOKE DATASET ---
print("Loading SMOKE subset:", len(smoke_paths), "images")

smoke_samples = load_images_multiproc(smoke_paths, num_workers=22)

print("Loaded:", len(smoke_samples))

# Quick preview (only filename + byte length)
for i in range(3):
    print(smoke_samples[i]["filename"], "| bytes:", len(smoke_samples[i]["bytes"]))


# In[5]:


# 🔥 CELL 4 — Build WebDataset shards

import webdataset as wds

for f in SMOKE_SHARDS.glob("smoke-*.tar"):
    f.unlink()

pattern = str(SMOKE_SHARDS / "smoke-%06d.tar")

with wds.ShardWriter(pattern, maxcount=3000) as sink:
    for i, item in enumerate(items):
        sink.write({
            "__key__": f"smoke-{i:08d}",
            "jpg": item["jpeg"],
            "cls": item["cls"],
            "fn": item["fn"],
        })

print("Shards written:", list(SMOKE_SHARDS.glob("*.tar")))


# In[ ]:


# 🔥 CELL 5 — Load SMOKE dataset via WebDataset


import io
from PIL import Image
import torch
import webdataset as wds
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def decode_sample(sample):
    img = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
    img = transform(img)
    wnid = sample["cls"]

    # Map WNID → class index
    cls_idx = sorted(smoke_split["indices"].keys()).index(wnid)

    return img, cls_idx


dataset = (
    wds.WebDataset(str(SMOKE_SHARDS / "smoke-%06d.tar"))
    .decode()
    .map(decode_sample)
)

loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

print("Dataloader ready.")


# In[ ]:


# 🔥 CELL 6 — Load a DINOv3 backbone


from dinov3 import load_backbone  # your function

device = "cuda" if torch.cuda.is_available() else "cpu"

model, meta, tag = load_backbone("dino_vit_small", device_str=device)
model.eval()

print("Backbone loaded:", tag)


# In[ ]:


# 


# In[ ]:





# In[ ]:




