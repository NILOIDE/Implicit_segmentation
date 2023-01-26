import os
from pathlib import Path

UKBB_IMGS_DIR = Path(r"D:\data")

def remove_unused_ims():
    ims = []
    segs = []
    bboxes = []
    for parent, subdir, files in os.walk(str(UKBB_IMGS_DIR)):
        for file in files:
            if file[-3:] == ".gz" and file[:2] not in {"sa", "la", "se"}:
                im_path = Path(parent) / file
                os.remove(str(im_path))
                continue
    return ims, segs, bboxes


if __name__ == '__main__':
    remove_unused_ims()
