import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from scipy.ndimage import rotate
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import nibabel as nib
import cv2

from data_loading.augmentations import TranslateCoords, RotateCoords, GammaShift

UKBB_IMGS_DIR = Path(r"C:\Users\nilst\Documents\Implicit_segmentation\data\ukbb")
UKBB_TEST_IMGS_DIR = Path(r"C:\Users\nilst\Documents\Implicit_segmentation\data\ukbb_test")


class SAX3D(Dataset):
    def __init__(self, load_dir=UKBB_IMGS_DIR, side_length=(128, 128, None), **kwargs):
        self.load_dir = load_dir
        self.shape = list(side_length)
        self.im_paths, self.seg_paths, self.bboxes = self.find_sax_images()
        self.augs = [GammaShift(gamma_lim=(0.6, 1.7))]
        self.num_aug_params = sum([a.num_parameters for a in self.augs])
        self.do_augment = True

    def find_sax_images(self):
        ims = []
        segs = []
        bboxes = []
        for parent, subdir, files in os.walk(str(self.load_dir)):
            for file in files:
                if file == "sa_ED.nii.gz":
                    im_path = Path(parent) / file
                    seg_path = Path(parent) / "seg_sa_ED.nii.gz"
                    seg = nib.load(seg_path).get_data()
                    arg = np.argwhere(seg > 0)
                    ims.append(im_path)
                    segs.append(seg_path)
                    bbox = (arg.min(0), arg.max(0))
                    bboxes.append(bbox)
                    if self.shape[2] is None or bbox[1][2]+1 - bbox[0][2] > self.shape[2]:
                        self.shape[2] = bbox[1][2]+1 - bbox[0][2]
        return ims, segs, bboxes

    def visualize(self):
        sh = 32
        canvas = np.zeros((0, sh * 12))
        for parent, subdir, files in os.walk(str(self.load_dir)):
            for file in files:
                if file == "sa_ED.nii.gz":
                    im_path = Path(parent) / file
                    seg_path = Path(parent) / "seg_sa_ED.nii.gz"
                    seg = nib.load(seg_path).get_data()
                    arg = np.argwhere(seg > 0)
                    bbox = (arg.min(0), arg.max(0))
                    im = nib.load(im_path).get_data()
                    im = im[bbox[0][0]: bbox[1][0],
                         bbox[0][1]: bbox[1][1],
                         bbox[0][2]: bbox[1][2]]
                    im = (im - im.min()) / (im.max() - im.min())
                    im = cv2.resize(im, (sh, sh))
                    zer = np.zeros((sh, sh * 12))
                    zer[:, :sh * im.shape[-1]] = np.concatenate([im[..., i] for i in range(im.shape[-1])], axis=1)
                    canvas = np.concatenate((canvas, zer), axis=0)
        return canvas

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        nii_img = nib.load(self.im_paths[idx])
        raw_data = nii_img.get_data()
        bbox_min, bbox_max = self.bboxes[idx]
        sub_vol = raw_data[bbox_min[0]: bbox_max[0],
                           bbox_min[1]: bbox_max[1],
                           bbox_min[2]: bbox_max[2]]
        min_, max_ = sub_vol.min(), sub_vol.max()
        sub_vol = (sub_vol - min_) / (max_ - min_)
        img = cv2.resize(sub_vol, self.shape[:2])
        img = torch.from_numpy(img)
        x, y, z = torch.meshgrid(torch.arange(img.shape[0], dtype=torch.float32),
                                 torch.arange(img.shape[1], dtype=torch.float32),
                                 torch.arange(img.shape[2], dtype=torch.float32))
        x = x / img.shape[0] - .5
        y = y / img.shape[1] - .5
        z = z / img.shape[2] - .5
        coords = torch.stack((x, y, z), dim=-1)
        return coords, img, idx


class SAX3D_Seg(SAX3D):

    def __getitem__(self, idx):
        nii_img = nib.load(self.im_paths[idx])
        im_data = nii_img.get_data()
        nii_seg = nib.load(self.seg_paths[idx])
        seg_data = nii_seg.get_data()
        bbox_min, bbox_max = self.bboxes[idx]

        sub_im = im_data[bbox_min[0]: bbox_max[0],
                         bbox_min[1]: bbox_max[1],
                         bbox_min[2]: bbox_max[2]]
        min_, max_ = sub_im.min(), sub_im.max()
        sub_im = (sub_im - min_) / (max_ - min_)
        img = cv2.resize(sub_im, self.shape[:2])
        img = torch.from_numpy(img)

        sub_seg = seg_data[bbox_min[0]: bbox_max[0],
                           bbox_min[1]: bbox_max[1],
                           bbox_min[2]: bbox_max[2]]
        seg = cv2.resize(sub_seg, self.shape[:2], interpolation=cv2.INTER_NEAREST)

        x, y, z = torch.meshgrid(torch.arange(img.shape[0], dtype=torch.float32),
                                 torch.arange(img.shape[1], dtype=torch.float32),
                                 torch.arange(img.shape[2], dtype=torch.float32))
        x = x / img.shape[0] - .5
        y = y / img.shape[1] - .5
        z = z / img.shape[2] - .5
        coords = torch.stack((x, y, z), dim=-1)

        aug_params = []
        if self.do_augment:
            data = {"coords": coords, "image": img, "seg": seg}
            for aug in self.augs:
                aug_params.extend(aug(data))
            coords, img, seg = data["coords"], data["image"], data["seg"]
        aug_params = torch.tensor(aug_params)

        return coords, img, idx, seg, aug_params


class SAX3D_padded(SAX3D):

    def __getitem__(self, idx):

        nii_img = nib.load(self.im_paths[idx])
        img_data = nii_img.get_data()
        bbox_min, bbox_max = self.bboxes[idx]
        if bbox_max[2] - bbox_min[2] < self.shape[2]:
            bbox_min[2] = max(0, bbox_max[2] - self.shape[2])
        if bbox_max[2] - bbox_min[2] < self.shape[2]:
            bbox_max[2] = min(img_data.shape[2], bbox_min[2] + self.shape[2])
        sub_vol = img_data[bbox_min[0]: bbox_max[0],
                           bbox_min[1]: bbox_max[1],
                           bbox_min[2]: bbox_max[2]]
        if sub_vol.shape[2] < self.shape[2]:
            pad = np.zeros((*sub_vol.shape[:2], self.shape[2] - sub_vol.shape[2]), dtype=np.float32)
            sub_vol = np.concatenate((sub_vol, pad), axis=2)
        min_, max_ = sub_vol.min(), sub_vol.max()
        sub_vol = (sub_vol - min_) / (max_ - min_)
        img = cv2.resize(sub_vol, self.shape[:2])
        img = torch.from_numpy(img)

        x, y, z = torch.meshgrid(torch.arange(img.shape[0], dtype=torch.float32),
                                 torch.arange(img.shape[1], dtype=torch.float32),
                                 torch.arange(img.shape[2], dtype=torch.float32))
        x = x / img.shape[0]
        y = y / img.shape[1]
        z = z / img.shape[2]
        coords = torch.stack((x, y, z), dim=-1)
        return coords, img, idx


class SAX3D_test(SAX3D):
    def __init__(self, load_dir=UKBB_TEST_IMGS_DIR, idx=0, **kwargs):
        super(SAX3D_test, self).__init__(load_dir, **kwargs)
        self.im_paths = self.im_paths[idx:idx+1]
        self.bboxes = self.bboxes[idx:idx+1]

        self.image = self.__getitem__(0)[1]
        self.do_augment = False


class SAX3D_Seg_test(SAX3D_Seg):
    def __init__(self, load_dir=UKBB_TEST_IMGS_DIR, idx=0, **kwargs):
        super(SAX3D_Seg_test, self).__init__(load_dir, **kwargs)
        self.im_paths = self.im_paths[idx:idx+1]
        self.bboxes = self.bboxes[idx:idx+1]

        self.image = self.__getitem__(0)[1]
        self.do_augment = False


class SAX3DCropped(Dataset):

    def __init__(self):
        self.max_shape = [0, 0, 0]
        self.im_paths = self.find_sax_images()

    def find_sax_images(self):
        ims = []
        for parent, subdir, files in os.walk(str(UKBB_IMGS_DIR)):
            for file in files:
                if file == "sa_ED.nii.gz":
                    path = Path(parent) / file
                    ims.append(path)
                    shape = nib.load(path).get_fdata().shape
                    if shape[0] > self.max_shape[0]:
                        self.max_shape[0] = shape[0]
                    if shape[1] > self.max_shape[1]:
                        self.max_shape[1] = shape[1]
                    if shape[2] > self.max_shape[2]:
                        self.max_shape[2] = shape[2]
        return ims

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):

        nii_img = nib.load(self.im_paths[idx])
        img = nii_img.get_data()
        canvas = np.zeros(self.max_shape, dtype=np.float32)
        offset = (np.array(canvas.shape) - np.array(img.shape)) // 2
        canvas[offset[0]:offset[0]+img.shape[0], offset[1]:offset[1]+img.shape[1], offset[2]:offset[2]+img.shape[2]] = img
        min_, max_ = canvas.min(), canvas.max()
        canvas = (canvas - min_) / (max_ - min_)
        # z = random.sample(range(0, canvas.shape[2]), 1)[0]

        img = canvas

        x, y, z = torch.meshgrid(torch.arange(img.shape[0], dtype=torch.float32),
                                 torch.arange(img.shape[1], dtype=torch.float32),
                                 torch.arange(img.shape[2], dtype=torch.float32),)
        # z = torch.full_like(x, z)

        x = x / img.shape[0]
        y = y / img.shape[1]
        z = z / canvas.shape[2]
        coords = torch.stack((x, y, z), dim=-1)
        return coords, img, idx


MNIST_IMGS_DIR = Path(r"/data/MNIST")
ROTATE_ANGLES = [45 * i for i in range(8)]

class MNIST3D(Dataset):
    def __init__(self):
        self.mnist_tensor_path = MNIST_IMGS_DIR.parent / "mnist_tensor.pt"
        self.images = self._create_time_dim()

        self.num_angles = len(ROTATE_ANGLES)
        self.num_samples = 28*28*4

    def _create_time_dim(self):
        try:
            raise FileNotFoundError
            new_images = torch.load(self.mnist_tensor_path)
        except FileNotFoundError:
            dataset = MNIST(root=str(IMGS_FILE), download=True)
            images = dataset.data.numpy()
            labels = dataset.train_labels.numpy()
            images = images[labels == 3][:32]
            new_images = torch.zeros((*images.shape, 8), dtype=torch.float32)
            for i in tqdm(range(images.shape[0]), desc="Processing time dimension."):
                im = images[i, ..., None]
                rotates = [torch.tensor(rotate(im, angle, reshape=False)) for angle in ROTATE_ANGLES]
                new_images[i] = torch.cat(rotates, dim=-1)
            new_images = new_images / 255
            torch.save(new_images, self.mnist_tensor_path)
        return new_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # y_idxs = torch.tensor(random.sample(range(0, img.shape[0]*self.num_samples), self.num_samples)) % img.shape[0]
        # x_idxs = torch.tensor(random.sample(range(0, img.shape[1]*self.num_samples), self.num_samples)) % img.shape[1]
        t = random.sample(range(0, self.num_angles), 1)[0]
        img = self.images[idx, ..., t]

        x, y = torch.meshgrid(torch.arange(img.shape[0], dtype=torch.float32), torch.arange(img.shape[1], dtype=torch.float32))
        t = torch.full_like(x, t)
        x = x / img.shape[0]
        y = y / img.shape[1]
        t = t / self.num_angles
        coords = torch.stack((x, y, t), dim=-1)
        return coords, img


class MNIST_SingleIM(Dataset):
    def __init__(self, im_idx=0):
        dataset = MNIST(root=str(MNIST_IMGS_DIR), download=True)
        self.images = dataset.data
        self.labels = dataset.train_labels
        self.images = self.images[self.labels == 3]
        self.image = self._create_time_dim(self.images[im_idx])
        self.num_angles = len(ROTATE_ANGLES)

    def set_image_num(self, im_num):
        self.image = self._create_time_dim(self.images[im_num])

    @staticmethod
    def _create_time_dim(image):
        image = image.numpy()
        im = image[..., None]
        rotates = [torch.tensor(rotate(im, angle, reshape=False)) for angle in ROTATE_ANGLES]
        new_im = torch.cat(rotates, dim=-1).to(torch.float32)
        return new_im / 255

    def __len__(self):
        return self.image.shape[2]

    def __getitem__(self, idx):
        img = self.image[..., idx]
        t = idx

        x, y = torch.meshgrid(torch.arange(img.shape[0], dtype=torch.float32), torch.arange(img.shape[1], dtype=torch.float32))
        t = torch.full_like(x, t)
        x = x / img.shape[0]
        y = y / img.shape[1]
        t = t / self.num_angles
        coords = torch.stack((x, y, t), dim=-1)
        return coords, img
