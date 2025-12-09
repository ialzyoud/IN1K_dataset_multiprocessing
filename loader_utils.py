#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path

def load_image_bytes(path):
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()
    return {
        "jpeg": raw,
        "cls": path.parent.name,
        "fn": path.name,
        "path": str(path),
    }

