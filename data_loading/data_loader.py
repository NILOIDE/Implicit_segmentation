from typing import Iterable, Dict, Any

import torch
from torch.utils.data import Dataset
import random
import numpy as np
import nibabel as nib
import cv2
from utils import draw_mask_to_image, find_sax_ED_images, find_sax_images, normalize_image, square_image

from data_loading.augmentations import TranslateCoords, RotateCoords, GammaShift


class AbstractDataset(Dataset):
    def __init__(self, load_dir, num_cases, case_start_idx=0,
                 side_length=(128, 128, -1), augmentations=(), **kwargs):
        self.load_dir = load_dir
        self.im_paths, self.seg_paths, self.bboxes = self.find_images(**kwargs)
        assert num_cases > 0
        if self.im_paths is not None:
            self.im_paths = self.im_paths[case_start_idx:case_start_idx+num_cases]
        if self.seg_paths is not None:
            self.seg_paths = self.seg_paths[case_start_idx:case_start_idx+num_cases]
        if self.bboxes is not None:
            self.bboxes = self.bboxes[case_start_idx:case_start_idx+num_cases]
        # If a side_length value was left as -1 by the user, use the max shape of that dimension instead
        self.out_shape = np.array(side_length)
        self.augs = self.parse_augmentations(augmentations)
        self.do_augment = self._do_augment
        self.num_aug_params = sum([a.num_parameters for a in self.augs])
        sample = self.__getitem__(0)
        self.sample_coords, self.sample_image, self.sample_seg = sample[0], sample[1], sample[3]

    def __len__(self):
        return len(self.im_paths)

    @property
    def _do_augment(self):
        return len(self.augs) > 0

    def find_images(self, **kwargs):
        raise NotImplementedError("This is abstract class. Implement your own.")

    @staticmethod
    def parse_augmentations(augs: Iterable[Dict[str, Any]]):
        name_2_class = {"translation": TranslateCoords, "rotation": RotateCoords, "gamma": GammaShift}
        aug_instances = []
        for aug in augs:
            n, params = list(aug.items())[0]
            try:
                class_name = name_2_class[n]
            except KeyError:
                raise KeyError(f'Provided augmentation name "{n}" is not in the default dictionary of augmentations.')
            aug_instances.append(class_name(**params))
        return aug_instances

    def visualize(self, im_path: str, seg_path: str = None, shape=(32, 32)):
        im = nib.load(im_path).get_data()
        im = (im - im.min()) / (im.max() - im.min())
        if seg_path is not None:
            seg = nib.load(seg_path).get_data()
            ims = [cv2.resize(draw_mask_to_image(im[..., i], seg[..., i]), shape) for i in range(im.shape[-1])]  # Converts slices into RGB
        else:
            im = cv2.resize(im, shape)
            ims = [im[..., i] for i in range(im.shape[-1])]
        canvas = np.stack(ims, axis=1)
        return canvas


class Seg3DCropped(AbstractDataset):
    def __init__(self, bbox_pad=5, **kwargs):
        super().__init__(**kwargs)
        self.bbox_pad = bbox_pad

    def __getitem__(self, idx):
        # Load image and seg data
        nii_img = nib.load(self.im_paths[idx])
        raw_im_data = nii_img.get_data()
        nii_seg = nib.load(self.seg_paths[idx])
        raw_seg_data = nii_seg.get_data()

        bbox_min, bbox_max = self.bboxes[idx]
        bbox_min = (bbox_min - self.bbox_pad).clip(min=0)
        bbox_max = (bbox_max + self.bbox_pad).clip(max=np.array(raw_im_data.shape))

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
        x = x / img.shape[0]
        y = y / img.shape[1]
        z = z / img.shape[2]
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


class Seg3DCropped_SAX(Seg3DCropped):
    """ IF you want to make one for some other data modality,
    creating a similar class with your own find_images method """
    def find_images(self, **kwargs):
        return find_sax_ED_images(self.load_dir, **kwargs)


class Seg3DCropped_SAX_test(Seg3DCropped_SAX):
    @property
    def _do_augment(self):
        return False


class Seg3DWholeImage(AbstractDataset):
    def __getitem__(self, idx):
        # Load image and seg data
        nii_img = nib.load(self.im_paths[idx])
        raw_im_data = nii_img.get_data()
        nii_seg = nib.load(self.seg_paths[idx])
        raw_seg_data = nii_seg.get_data()

        # Crop the image into a square based on which side is the longest
        cropped_im, cropped_seg = square_image(raw_im_data, raw_seg_data)
        # Normalize image to range [0, 1]
        img = normalize_image(cropped_im)
        seg = cropped_seg

        img = cv2.resize(img, self.out_shape[:2])
        seg = cv2.resize(seg, self.out_shape[:2], interpolation=cv2.INTER_NEAREST)

        x, y, z = torch.meshgrid(torch.arange(img.shape[0], dtype=torch.float32),
                                 torch.arange(img.shape[1], dtype=torch.float32),
                                 torch.arange(img.shape[2], dtype=torch.float32))
        x = x / img.shape[0]
        y = y / img.shape[1]
        z = z / img.shape[2]
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


class Seg3DWholeImage_SAX(Seg3DWholeImage):
    """ IF you want to make one for some other data modality,
    creating a similar class with your own find_images method """
    def find_images(self, **kwargs):
        return find_sax_ED_images(self.load_dir, **kwargs)


class Seg3DWholeImage_SAX_test(Seg3DWholeImage_SAX):
    @property
    def _do_augment(self):
        return False


class Seg4DWholeImage(AbstractDataset):
    def __getitem__(self, idx):
        # Load image and seg data
        nii_img = nib.load(self.im_paths[idx])
        nii_seg = nib.load(self.seg_paths[idx])
        raw_shape = nii_img.shape

        # Take only one time frame of the series
        t = random.randint(0, raw_shape[-1] - 1)
        frame_im_data = nii_img.dataobj[..., t]
        frame_seg_data = nii_seg.dataobj[..., t].astype(np.uint8)
        t = t / raw_shape[-1]

        # Crop the image into a square based on which side is the longest
        cropped_im, cropped_seg = square_image(frame_im_data, frame_seg_data)
        # Normalize image to range [0, 1]
        img = normalize_image(cropped_im)
        seg = cropped_seg

        img = cv2.resize(img, self.out_shape[:2])
        seg = cv2.resize(seg, self.out_shape[:2], interpolation=cv2.INTER_NEAREST)

        # torch.meshgrid has a different behaviour than np.meshgrid
        x, y, z = torch.meshgrid(torch.arange(img.shape[0], dtype=torch.float32),
                                 torch.arange(img.shape[1], dtype=torch.float32),
                                 torch.arange(img.shape[2], dtype=torch.float32))
        x = x / img.shape[0]
        y = y / img.shape[1]
        z = z / img.shape[2]
        t = torch.full_like(x, t)
        coords = torch.stack((x, y, z, t), dim=-1).numpy()

        aug_params = []
        if self.do_augment:
            data = {"coords": coords, "image": img, "seg": seg}
            for aug in self.augs:
                aug_params.extend(aug(data))
            coords, img, seg = data["coords"], data["image"], data["seg"]
        aug_params = torch.tensor(aug_params)

        coords = torch.from_numpy(coords).to(torch.float32)
        img = torch.from_numpy(img).to(torch.float32)
        seg = torch.from_numpy(seg).to(torch.uint8)
        idx = torch.tensor(idx).to(torch.long)
        return coords, img, idx, seg, aug_params


class Seg4DWholeImage_SAX(Seg4DWholeImage):
    """ IF you want to make one for some other data modality,
    creating a similar class with your own find_images method """
    def find_images(self, **kwargs):
        return find_sax_images(self.load_dir, **kwargs)


class Seg4DWholeImage_SAX_test(Seg4DWholeImage_SAX):
    @property
    def _do_augment(self):
        return False
