## IFQA - Official Pytorch Implementation [[Project Page]](https://sites.google.com/view/vcl-lab/publications/international-conference/ifqa_wacv23)

<img src="./docs/teaser.png" width="400">
 
> **IFQA: Interpretable Face Quality Assessment**<br>
> Byungho Jo, Donghyeon Cho, In Kyu Park, Sungeun Hong<br>
> In WACV 2023

> Paper: [https://arxiv.org/abs/2211.07077](https://arxiv.org/abs/2211.07077) <br>

> **Abstract:** *Existing face restoration models have relied on general assessment metrics that do not consider the characteristics of facial regions.
Recent works have therefore assessed their methods using human studies, which is not scalable and involves significant effort. This paper proposes a novel face-centric metric based on an adversarial framework where a generator simulates face restoration and a discriminator assesses image quality. Specifically, our per-pixel discriminator enables interpretable evaluation that cannot be provided by traditional metrics. Moreover, our metric emphasizes facial primary regions considering that even minor changes to the eyes, nose, and mouth significantly affect human cognition. Our face-oriented metric consistently surpasses existing general or facial image quality assessment metrics by impressive margins. We demonstrate the generalizability of the proposed strategy in various architectural designs and challenging scenarios. Interestingly, we find that our IFQA can lead to performance improvement as an objective function.*

## Additional material
- [IFQA pre-trained model](https://drive.google.com/file/d/1aHxF39Mdg4R2dFiF_yx8HsJHy9lJaEZP/view?usp=sharing)
- [Video]

## Requirements
* 64-bit Python 3.7 and PyTorch 1.7.0 (or later). See https://pytorch.org for PyTorch install instructions.
* Albumentations. See https://albumentations.ai/ for Albumentations install instructions.

## Usage
IFQA is designed for evaluating the realness of faces. IFQA produces score maps of each pixel and we apply average to get final score.

You can produce quality scores using `test.py`. For example:
```.bash
# A single face image input.
python test.py --path=./faces0.png

# All images within a directory.
python test.py --path=./testA
```

## Citation

```
@inproceedings{Jo_2023_WACV,
  author = {Byungho, Jo and Cho, Donghyeon and Park, In Kyu and Hong, Sungeun},
  title = {IFQA: Interpretable Face Quality Assessment},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month = {January},
  year = {2023},
  pages = {-}
}
```

## Acknowledgements

We thank Eunkyung Jo for helpful feedback on human study design and Jaejun Yoo for constructive comments on various experimental protocols.
