# GS-Checker

This is the official implementation of [GS-Checker: Tampering Localization for 3D Gaussian Splatting](https://arxiv.org/abs/2511.20354) [AAAI 2026].


# Installation

```bash
git clone git@github.com:haolianghan/GS-Checker.git
```
or
```bash
git clone https://github.com/haolianghan/GS-Checker.git
```

```bash
cd GS-Checker
```

Then install the dependencies:
```bash
conda env create --file environment.yml
conda activate checker
```

Download SAFIRE weights:
download the weights from [Google Drive Link](https://drive.google.com/drive/folders/1NRxep2G42OnVwCR9sGdf1iPqhCUrGmv2) and put them in the `./SAFIRE` folder.


## Data

Our dataset will be released soon.

## Tampering localization
```bash
bash run.sh
```
## Rendering
you can render the 3D tampering localization results by running the following command:
```bash
python render.py -m <path to the 3DGS model> --scene_name <name of the original scene>
```

# Citation
If you find our paper useful for your work please cite:

```BibTex
@inproceedings{han2026gs,
      title={GS-Checker: Tampering Localization for 3D Gaussian Splatting}, 
      author={Haoliang Han and Ziyuan Luo and Jun Qi and Anderson Rocha and Renjie Wan},
      booktitle={AAAI},
      year={2026}
}
```
