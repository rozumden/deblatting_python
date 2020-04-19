# Deblatting in Python
Python implementation of deblatting (*debl*urring and m*atting*). We assume the following formation model of input image `I`:

<img src="imgs/fmo_model.png" width="500">

<img src="imgs/fmo_formation_ex.png" width="500">

This repository contains various algorithms to estimate unknown `F, M` and `H`. The background `B` is assumed to be given, for example by a median of 3-5 previous video frames.

Using
-----
### Estimate F, M, H
The most general case is when everything needs to be estimated. In this case it is advisable to provide at least an approximate object size, e.g. by `M0 = np.ones([diameter]*2)`. Otherwise an object of `I.shape` is being estimated, which can be slow. For speed up, inputs can be cropped to the area near the object of interest, or `Hmask` can be provided as a binary mask in the input image where the object is present.
```python
H, F, M = estimateFMH(I, B, M0)
```

Publications
------------
This repository contains the implementation of the followings papers:

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
