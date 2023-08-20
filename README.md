# FOCAL

An official implementation code for paper "Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering"

<p align='center'>  
  <img src='https://github.com/HighwayWu/FOCAL/blob/main/imgs/chart.jpg' width='850'/>
</p>
<p align='center'>  
  <em>IoU performance comparison with SOTAs on six cross-domain testsets.</em>
</p>

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Usage](#usage)
- [Citation](#citation)


## Background
Image forgery detection aims to detect and locate forged regions in an image. Most existing forgery detection algorithms formulate classification problems to classify pixels into forged or pristine. However, the definition of forged and pristine pixels is only relative within one single image, e.g., a forged region in image A is actually a pristine one in its source image B (splicing forgery). Such a relative definition has been severely overlooked by existing methods, which unnecessarily mix forged (pristine) regions across different images into the same category. 

<p align='center'>  
  <img src='https://github.com/HighwayWu/FOCAL/blob/main/imgs/head_fig.jpg' width='850'/>
</p>
<p align='center'>  
  <em> First row: input pristine and forged images. Second row: forgery masks, where pristine (α1, α2 and α3) and forged (β1 and β2) regions are labeled black and white, respectively.</em>
</p>

To resolve this dilemma, we propose the FOrensic ContrAstive cLustering (FOCAL) method, a novel, simple yet very effective paradigm based on contrastive learning and unsupervised clustering for the image forgery detection. Specifically, FOCAL 1) utilizes pixel-level contrastive learning to supervise the high-level forensic feature extraction in an image-by-image manner, explicitly reflecting the above relative definition; 2) employs an on-the-fly unsupervised clustering algorithm (instead of a trained one) to cluster the learned features into forged/pristine categories, further suppressing the cross-image influence from training data; and 3) allows to further boost the detection performance via simple feature-level concatenation without the need of retraining.

<p align='center'>  
  <img src='https://github.com/HighwayWu/FOCAL/blob/main/imgs/framework.jpg' width='850'/>
</p>
<p align='center'>  
  <em> Our proposed FOCAL framework, which utilizes contrastive learning to supervise the training phase, while employing an unsupervised clustering algorithm in the testing phase.</em>
</p>

Extensive experimental results over six public testing datasets demonstrate that our proposed FOCAL significantly outperforms the state-of-the-art competing algorithms by big margins: +24.3% on Coverage, +18.6% on Columbia, +17.5% on FF++, +14.2 on MISD, +13.5% on CASIA and +10.3% on NIST in terms of IoU. The paradigm of FOCAL could bring fresh insights and serve as a novel benchmark for the image forgery detection task.

<p align='center'>  
  <img src='https://github.com/HighwayWu/FOCAL/blob/main/imgs/cmp.jpg' width='850'/>
</p>
<p align='center'>  
  <em> Qualitative comparison of forgery detection results on some representative testing images.</em>
</p>

## Dependency
- torch 1.9.0
- scikit-learn 1.2.1
- torch_kmeans 0.2.0

## Usage

- For training:
```bash
python main.py --type='train'
```

- For testing:
```bash
python main.py --type='test_single'
```
FOCAL will detect the images in the `demo/input/` and save the results in the `demo/output/` directory.

- For prepare the training/test datasets:
```bash
python main.py --type='flist' --path_input 'demo/input/' --path_gt 'demo/gt/' --nickname 'demo'
```

**Note: The pretrained FOCAL can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12ayIO9PU4wvqWqniT3KtH8tCvrZ-M-zd?usp=sharing).**

## Citation

If you use this code for your research, please citing the reference:
```
@article{focal,
  title={Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering},
  author={H. Wu and Y. Chen and J. Zhou},
  year={2023}
}
```
