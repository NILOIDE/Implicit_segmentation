# NISF: Neural Implicit Segmentation Functions

This is the official repository for the MICCAI 2023 paper submission: "NISF: Neural Implicit Segmentation Functions"


### NISFs are able to segment shapes at arbitrary resolutions:

![Alt text](images/Interpolation_diagram.png?raw=true "Arbitrary segmentation resolution")


### NISFs can create smooth segmentations along arbitrary image planes:

![Alt text](images/4CH_segmentation.png?raw=true "Arbitrary plane segmentation")


### NISFs implicitly model priors, allowing them to segment regions not available in the image volume:

![Alt text](images/SAX_predictions_z_holdout.png?raw=true "Segmentation of regions not available in image volume")


## Citation and Contribution

Please cite this work if any of our code or ideas are helpful for your research.

```
@inproceedings{stolt2023nisf,
  title={NISF: Neural Implicit Segmentation Functions},
  author={Stolt-Ans{\'o}, Nil and McGinnis, Julian and Pan, Jiazhen and Hammernik, Kerstin and Rueckert, Daniel},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={734--744},
  year={2023},
  organization={Springer}
}
```
