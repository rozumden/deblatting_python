# Deblatting in Python
Python implementation of deblatting (*debl*urring and m*atting*). We assume the following formation model of input image `I`:

<img src="imgs/fmo_model.png" width="500">

<img src="imgs/fmo_formation_ex.png" width="500">

This repository contains various algorithms to estimate unknown `F, M` and `H`. The background `B` is assumed to be given, for example by a median of 3-5 previous video frames.

Dependencies
------------
The library is written in Python. The following packages are required: `numpy, scipy, skimage`. For visualization purposes, also `opencv` is recommended.

Using
-----
All parameters are explained in `Params()` class. Provided script `run.py` shows examples how to use the methods. For GPU version in PyTorch check `run_gpu.py`.
### Estimate F, M, H
The most general case is when everything needs to be estimated. In this case it is advisable to provide at least an approximate object size, e.g. by `M0 = np.ones([diameter]*2)`. Otherwise an object of `I.shape` is being estimated, which can be slow. For speed up, inputs can be cropped to the area near the object of interest, or `Hmask` can be provided as a binary mask in the input image where the object is present.
```python
H, F, M = estimateFMH(I, B, M0)
```

<img src="imgs/fmh.gif" width="500">
<img src="imgs/syn.gif" width="500">
<img src="imgs/vol.gif" width="500">

### Estimate F, M given H
```python
F, M = estimateFM(I, B, H)
```

<img src="imgs/estFM.gif" width="500">

### Estimate H given F, M
```python
H = estimateH(I, B, M, F)
```

<img src="imgs/estH.gif" width="500">

### Estimate (F_i, M_i) given (H_i)
Piece-wise deblatting. Single `M0 = np.ones([diameter]*2)` vs varying `Ms0 = np.ones([diameter]*2 +[ns])`, where `ns` is the number of splits.
```python
Fs, Ms = estimateFM_pw(I, B, Hs, M0)
```

<img src="imgs/pw.gif" width="500">

```python
Fs, Ms = estimateFM_pw(I, B, Hs, Ms0)
```
<img src="imgs/pw_ms.gif" width="500">

### GPU version in PyTorch
Input data must be 4-D in format (K,C,W,H), where K is the number of images in a batch, C is the number of channels, and W/H are image width and height. The output will be in the same format, and each image in the batch is processed in parallel.

```python
H, F, M = estimateFMH_gpu(I, B, M0)
```

```python
F, M = estimateFM_gpu(I, B, H)
```

```python
H = estimateH_gpu(I, B, M, F)
```

Benchmarking
------------
Evaluation on real-world data can be performed on the [FMO deblurring benchmark](https://github.com/rozumden/fmo-deblurring-benchmark).


Publications
------------
This repository contains implementation of the following publications:

```bibtex
@inproceedings{Rozumnyi-et-al-CVPR-2020,
  author = {Denys Rozumnyi and Jan Kotera and Filip Sroubek and Jiri Matas},
  title = {Sub-frame Appearance and 6D Pose Estimation of Fast Moving Objects},
  booktitle = {CVPR},
  address = {Seattle, Washington, USA},
  month = jun,
  year = {2020}
}
@inproceedings{Kotera-et-al-ICCVW-2019,
  author = {Jan Kotera and Denys Rozumnyi and Filip Sroubek and Jiri Matas},
  title = {Intra-frame Object Tracking by Deblatting},
  booktitle = {Internatioal Conference on Computer Vision Workshop (ICCVW), 
  Visual Object Tracking Challenge Workshop, 2019},
  address = {Seoul, South Korea},
  month = oct,
  year = {2019}
}
```
Other related papers:
```bibtex
@inproceedings{Rozumnyi-et-al-GCPR-2019,
  author = {Denys Rozumnyi and Jan Kotera and Filip Sroubek and Jiri Matas},
  title = {Non-Causal Tracking by Deblatting},
  booktitle = {GCPR},
  address = {Dortmund, Germany},
  month = sep,
  year = {2019}
}
@inproceedings{Rozumnyi-et-al-CVPR-2017,
  author = {Denys Rozumnyi and Jan Kotera and Filip Sroubek and Lukas Novotny and Jiri Matas},
  title = {The World of Fast Moving Objects},
  booktitle = {CVPR},
  address = {Honolulu, Hawaii, USA},
  month = jul,
  year = {2017}
}
```
