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

UKBB_IMGS_DIR = Path(r"C:\Users\nilst\Documents\Implicit_segmentation\data\ukbb_small")
UKBB_TEST_IMGS_DIR = Path(r"C:\Users\nilst\Documents\Implicit_segmentation\data\ukbb_test")


class SAX3D(Dataset):
    def __init__(self, load_dir=UKBB_IMGS_DIR, side_length=(128, 128, -1), heart_pad=5, **kwargs):
        self.load_dir = load_dir
        self.heart_pad = heart_pad
        self.max_shape = np.full((len(side_length)), -1, dtype=int)
        self.im_paths, self.seg_paths, self.bboxes = self.find_sax_images()
        # If a side_length value was left as -1 by the user, use the max shape of that dimension instead
        self.out_shape = np.array([side_length[0], side_length[1], self.max_shape[2]])
        self.augs = []#[TranslateCoords(x_lim=0.05, y_lim=0.05)]
        self.num_aug_params = sum([a.num_parameters for a in self.augs])
        self.do_augment = True

    def __len__(self):
        return len(self.im_paths)

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
                    if self.max_shape[0] < 0 or bbox[1][0] + 1 - bbox[0][0] > self.max_shape[0]:
                        self.max_shape[0] = bbox[1][0] + 1 - bbox[0][0]
                    if self.max_shape[1] < 0 or bbox[1][1] + 1 - bbox[0][1] > self.max_shape[1]:
                        self.max_shape[1] = bbox[1][1] + 1 - bbox[0][1]
                    if self.max_shape[2] < 0 or bbox[1][2] + 1 - bbox[0][2] > self.max_shape[2]:
                        self.max_shape[2] = bbox[1][2] + 1 - bbox[0][2]
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

    def __getitem__(self, idx):
        # Load image data
        nii_img = nib.load(self.im_paths[idx])
        raw_data = nii_img.get_data()
        # Extract section of image based on precomputed bounding box
        bbox_min, bbox_max = self.bboxes[idx]
        pad = self.heart_pad
        sub_vol = raw_data[bbox_min[0]-pad: bbox_max[0]+pad,
                           bbox_min[1]-pad: bbox_max[1]+pad,
                           bbox_min[2]: bbox_max[2]]
        min_, max_ = sub_vol.min(), sub_vol.max()
        sub_vol = (sub_vol - min_) / (max_ - min_)
        img = cv2.resize(sub_vol, self.out_shape[:2])

        x, y, z = np.meshgrid(np.arange(img.shape[0], dtype=float),
                              np.arange(img.shape[1], dtype=float),
                              np.arange(img.shape[2], dtype=float))
        x = x / img.shape[0] - .5
        y = y / img.shape[1] - .5
        z = z / img.shape[2] - .5
        coords = torch.stack((x, y, z), dim=-1)

        aug_params = []
        if self.do_augment:
            data = {"coords": coords, "image": img}
            for aug in self.augs:
                aug_params.extend(aug(data))
            coords, img, seg = data["coords"], data["image"], data["seg"]
        aug_params = torch.tensor(aug_params)

        coords = torch.from_numpy(coords).to(torch.float32)
        img = torch.from_numpy(img).to(torch.float32)
        return coords, img, idx, aug_params


class SAX3D_test(SAX3D):
    def __init__(self, load_dir=UKBB_TEST_IMGS_DIR, idx=0, **kwargs):
        super(SAX3D_test, self).__init__(load_dir, **kwargs)
        self.im_paths = [self.im_paths[idx] for _ in range(kwargs["val_max_epochs"])]
        self.bboxes = [self.bboxes[idx] for _ in range(kwargs["val_max_epochs"])]

        self._batch = None
        self.do_augment = False
        self.image = self.__getitem__(0)[1]

    def __getitem__(self, idx):
        if self._batch is None:
            batch = super().__getitem__(idx)
            self.batch_ = (item.cuda() for item in batch)
        return self._batch


class SAX3D_Seg(SAX3D):

    def __getitem__(self, idx):
        # Load image and seg data
        nii_img = nib.load(self.im_paths[idx])
        raw_im_data = nii_img.get_data()
        nii_seg = nib.load(self.seg_paths[idx])
        raw_seg_data = nii_seg.get_data()

        bbox_min, bbox_max = self.bboxes[idx]
        bbox_min = (bbox_min - self.heart_pad).clip(min=0)
        bbox_max = (bbox_max + self.heart_pad).clip(max=np.array(raw_im_data.shape))

        sub_im = raw_im_data[bbox_min[0]: bbox_max[0],
                             bbox_min[1]: bbox_max[1],
                             bbox_min[2]: bbox_max[2]]
        min_, max_ = sub_im.min(), sub_im.max()
        sub_im = (sub_im - min_) / (max_ - min_)
        img = cv2.resize(sub_im, self.out_shape[:2])

        sub_seg = raw_seg_data[bbox_min[0]: bbox_max[0],
                               bbox_min[1]: bbox_max[1],
                               bbox_min[2]: bbox_max[2]]
        seg = cv2.resize(sub_seg, self.out_shape[:2], interpolation=cv2.INTER_NEAREST)

        x, y, z = np.meshgrid(np.arange(img.shape[0], dtype=float),
                              np.arange(img.shape[1], dtype=float),
                              np.arange(img.shape[2], dtype=float))
        x = x / img.shape[0] - .5
        y = y / img.shape[1] - .5
        z = z / img.shape[2] - .5
        coords = np.stack((x, y, z), axis=-1)

        aug_params = []
        if self.do_augment:
            data = {"coords": coords, "image": img, "seg": seg}
            for aug in self.augs:
                aug_params.extend(aug(data))
            coords, img, seg = data["coords"], data["image"], data["seg"]
        aug_params = torch.tensor(aug_params)

        coords = torch.from_numpy(coords).to(torch.float32)
        img = torch.from_numpy(img).to(torch.float32)
        seg = torch.from_numpy(seg)
        idx = torch.tensor(idx)
        return coords, img, idx, seg, aug_params


class SAX3D_Seg_test(SAX3D_Seg):
    def __init__(self, load_dir=UKBB_TEST_IMGS_DIR, idx=0, **kwargs):
        super(SAX3D_Seg_test, self).__init__(load_dir, **kwargs)
        self.im_paths = self.im_paths[idx:idx+1]
        self.bboxes = self.bboxes[idx:idx+1]
        self._len = kwargs["val_max_epochs"]

        self._batch = None
        self.do_augment = False
        self.image = self.__getitem__(0)[1]

    def __getitem__(self, idx):
        if self._batch is None:
            batch = super().__getitem__(idx)
            self._batch = tuple(item.cuda() for item in batch)
        return self._batch


class SAX3D_Seg_WholeImage(SAX3D_Seg):
    def __getitem__(self, idx):
        # Load image and seg data
        nii_img = nib.load(self.im_paths[idx])
        raw_im_data = nii_img.get_data()
        nii_seg = nib.load(self.seg_paths[idx])
        raw_seg_data = nii_seg.get_data()

        max_dim = np.argmax(raw_im_data.shape)
        if max_dim == 0:
            start_idx = (raw_im_data.shape[0] - raw_im_data.shape[1]) // 2
            cropped_im = raw_im_data[start_idx: start_idx + raw_im_data.shape[1]]
            cropped_seg = raw_seg_data[start_idx: start_idx + raw_seg_data.shape[1]]
        elif max_dim == 1:
            start_idx = (raw_im_data.shape[1] - raw_im_data.shape[0]) // 2
            cropped_im = raw_im_data[:, start_idx: start_idx + raw_im_data.shape[0]]
            cropped_seg = raw_seg_data[:, start_idx: start_idx + raw_seg_data.shape[0]]
        else:
            raise ValueError
        assert cropped_im.shape[0] == cropped_im.shape[1]

        min_, max_ = cropped_im.min(), cropped_im.max()
        img = (cropped_im - min_) / (max_ - min_)
        seg = cropped_seg
        img = cv2.resize(img, self.out_shape[:2])

        seg = cv2.resize(seg, self.out_shape[:2], interpolation=cv2.INTER_NEAREST)

        x, y, z = torch.meshgrid(torch.arange(img.shape[0], dtype=torch.float32),
                                 torch.arange(img.shape[1], dtype=torch.float32),
                                 torch.arange(img.shape[2], dtype=torch.float32))
        x = x / img.shape[0] - .5
        y = y / img.shape[1] - .5
        z = z / img.shape[2] - .5
        coords = torch.stack((x, y, z), dim=-1).numpy()

        aug_params = []
        if self.do_augment:
            data = {"coords": coords, "image": img, "seg": seg}
            for aug in self.augs:
                aug_params.extend(aug(data))
            coords, img, seg = data["coords"], data["image"], data["seg"]
        aug_params = torch.tensor(aug_params)

        coords = torch.from_numpy(coords).to(torch.float32)
        img = torch.from_numpy(img).to(torch.float32)
        seg = torch.from_numpy(seg)
        idx = torch.tensor(idx)
        return coords, img, idx, seg, aug_params


class SAX3D_Seg_WholeImage_test(SAX3D_Seg_WholeImage):
    def __init__(self, load_dir=UKBB_TEST_IMGS_DIR, idx=0, **kwargs):
        super(SAX3D_Seg_WholeImage, self).__init__(load_dir, **kwargs)
        self.im_paths = self.im_paths[idx:idx+1]
        self.bboxes = self.bboxes[idx:idx+1]
        self._len = kwargs["val_max_epochs"]

        self._batch = None
        self.do_augment = False
        self.image = self.__getitem__(0)[1]

    def __getitem__(self, idx):
        if self._batch is None:
            batch = super().__getitem__(idx)
            self._batch = tuple(item.cuda() for item in batch)
        return self._batch


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
