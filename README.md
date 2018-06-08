# PatchCamelyon (PCam)
_That which is measured, improves._ - Karl Pearson

The PatchCamelyon benchmark is a new and challenging image classification dataset. It consists of 327.680 color images (96 x 96px) extracted from histopathologic scans of lymph node sections. Each image is annoted with a binary label indicating presence of metastatic tissue. PCam provides a new benchmark for machine learning models: bigger than CIFAR10, smaller than imagenet, trainable on a single GPU.

![PCam example images. Green boxes indicate positive labels.](https://github.com/basveeling/pcam/blob/master/pcam.jpg)
*Example images from PCam. Green boxes indicate tumor tissue in center region, which dictates a positive label.*

<details><summary>Table of Contents</summary><p>

* [Why PCam](#why-pcam)
* [Download](#download)
* [Details](#details)
* [Usage and Tips](#usage-and-tips)
* [Benchmark](#benchmark)
* [Visualization](#visualization)
* [Contributing](#contributing)
* [Contact](#contact)
* [Citing PCam](#citing-pcam)
* [License](#license)
</p></details><p></p>

## Why PCam
Fundamental machine learning advancements are predominantly evaluated on straight-forward natural-image classification datasets. Think MNIST, CIFAR, SVHN. Medical imaging is becoming one of the major applications of ML and we believe it deserves a spot on the list of _go-to_ ML datasets. Both to challenge future work, and to steer developments into directions that are beneficial for this domain.

We think PCam can play a role in this. It packs the clinically-relevant task of metastasis detection into a straight-forward binary image classification task, akin to CIFAR-10 and MNIST. Models can easily be trained on a single GPU in a couple hours, and achieve competitive scores in the Camelyon16 tasks of tumor detection and WSI diagnosis. Furthermore, the balance between task-difficulty and tractability makes it a prime suspect for fundamental machine learning research on topics as active learning, model uncertainty and explainability.


## Download
The data is stored in gzipped HDF5 files and can be downloaded using the following links. Each set consist of a data and target file. An additional meta csv file is provided which describes from which Camelyon16 slide the patches were extracted from, but this information is not used in training for or evaluating the benchmark. Please report any downloading problems via a github issue.

| Name  | Content | Size | Link | MD5 Checksum|
| --- | --- |--- | --- |--- |
| `camelyonpatch_level_2_split_train_x.h5.gz` | training images | 6.1 GB | [Download](https://drive.google.com/uc?export=download&id=1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2)|`1571f514728f59376b705fc836ff4b63`|
| `camelyonpatch_level_2_split_train_y.h5.gz` | training labels | 21 KB | [Download](https://drive.google.com/uc?export=download&id=1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG)|`35c2d7259d906cfc8143347bb8e05be7`|
| `camelyonpatch_level_2_split_valid_x.h5.gz` | valid images | 0.8 GB | [Download](https://drive.google.com/uc?export=download&id=1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3)|`d8c2d60d490dbd479f8199bdfa0cf6ec`|
| `camelyonpatch_level_2_split_valid_y.h5.gz` | valid labels | 3.0 KB | [Download](https://drive.google.com/uc?export=download&id=1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO)|`60a7035772fbdb7f34eb86d4420cf66a`|
| `camelyonpatch_level_2_split_test_x.h5.gz`  | test images  | 0.8 GB | [Download](https://drive.google.com/uc?export=download&id=1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_)|`d5b63470df7cfa627aeec8b9dc0c066e`|
| `camelyonpatch_level_2_split_test_y.h5.gz`  | test labels  | 3.0 KB | [Download](https://drive.google.com/uc?export=download&id=17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP)|`2b85f58b927af9964a4c15b8f7e8f179`|
| `camelyonpatch_level_2_split_train_meta.csv` | training meta |  | [Download](https://drive.google.com/uc?export=download&id=1XoaGG3ek26YLFvGzmkKeOz54INW0fruR)|`5a3dd671e465cfd74b5b822125e65b0a`|
| `camelyonpatch_level_2_split_valid_meta.csv` | valid meta | | [Download](https://drive.google.com/uc?export=download&id=16hJfGFCZEcvR3lr38v3XCaD5iH1Bnclg)|`3455fd69135b66734e1008f3af684566`|
| `camelyonpatch_level_2_split_test_meta.csv`  | test meta |  | [Download](https://drive.google.com/uc?export=download&id=19tj7fBlQQrd4DapCjhZrom_fA4QlHqN4)|`67589e00a4a37ec317f2d1932c7502ca`|



## Usage and Tips
### Keras Example
[General dataloader for keras](https://github.com/basveeling/pcam/blob/master/keras/dataset/pcam.py)

```python
from keras.utils import HDF5Matrix
from keras.preprocessing.image import ImageDataGenerator

x_train = HDF5Matrix('camelyonpatch_level_2_split_train_x.h5', 'x')
y_train = HDF5Matrix('camelyonpatch_level_2_split_train_y.h5', 'y')

datagen = ImageDataGenerator(
              preprocessing_function=lambda x: x/255.,
              width_shift_range=4,  # randomly shift images horizontally
              height_shift_range=4,  # randomly shift images vertically 
              horizontal_flip=True,  # randomly flip images
              vertical_flip=True)  # randomly flip images
              
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) // batch_size
                    epochs=1024,
                    )
```

## Details
### Numbers
The dataset is divided into a training set of 262.144 (2^18) examples, and a validation and test set both of 32.768 (2^15) examples. There is no overlap in WSIs between the splits, and all splits have a 50/50 balance between positive and negative examples.

### Labeling
A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable the design of fully-convolutional models that do not use any zero-padding, to ensure consistent behavior when applied to a whole-slide image. This is however not a requirement for the PCam benchmark.

### Patch selection 
PCam is derived from the Camelyon16 Challenge [2], which contains 400 H\&E stained WSIs of sentinel lymph node sections. The slides were acquired and digitized at 2 different centers  using a 40x objective (resultant pixel resolution of 0.243 microns). We undersample this at 10x to increase the field of view.
We follow the train/test split from the Camelyon16 challenge [2], and further hold-out 20% of the train WSIs for the validation set. To prevent selecting background patches, slides are converted to HSV, blurred, and patches filtered out if maximum pixel saturation lies below 0.07 (which was validated to not throw out tumor data in the training set).
The patch-based dataset is sampled by iteratively choosing a WSI and selecting a positive or negative patch with probability _p_. Patches are rejected following a stochastic hard-negative mining scheme with a small CNN, and _p_ is adjusted to retain a balance close to 50/50.

### Statistics
_Coming soon_

## Contact
For problems and questions not fit for a github issue, please email [Bas Veeling](mailto:basveeling+pcam@gmail.com).
## Citing PCam
If you use PCam in a scientific publication, we would appreciate references to the following paper:

**[1] <TODO> [arXiv:<todo>](http://arxiv.org/abs/<todo>)**

A citation of the original Camelyon16 dataset paper is appreciated as well:

**[2] Ehteshami Bejnordi et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA: The Journal of the American Medical Association, 318(22), 2199–2210. [doi:jama.2017.14585](https://doi.org/10.1001/jama.2017.14585)**


Biblatex entry:
```latex
Coming...
```

<!-- [Who is citing PCam?](https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=pcam&btnG=&oq=fas) -->


## Benchmark
| Name  | Reference | Augmentations | Acc | AUC|  NLL | FROC* |
| --- | --- | --- | --- | --- | --- | --- |
| GDensenet | TODO ARXIV | Following Liu et al. | 89.8 | 96.3 |  0.260 |75.8 (64.3, 87.2)|
| [Add yours](https://github.com/basveeling/pcam/compare) | |

\* Performance on Camelyon16 tumor detection task, not part of the PCam benchmark.


## Contributing
Contributions with example scripts for other frameworks are welcome!

## License
The data is provided under the [CC0 License](https://choosealicense.com/licenses/cc0-1.0/), following the license of Camelyon16.

The rest of this repository is under the [MIT License](https://choosealicense.com/licenses/mit/)

## Acknowledgements
* Babak Ehteshami Bejnordi, Geert Litjens, Jeroen van der Laak for their input on the configuration of this dataset.
* README derived from [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
