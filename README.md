# PnP-CoSMo

<p align="center">
  <img src="docs/GraphicalAbstract.png"  width="800" alt="Overview of PnP-CoSMo, our two-stage approach to guided reconstruction. (a) The first stage learns a content/style model of two-contrast MR image data offline. We assume that the two image domains X1 and X2 can be decomposed into a shared content domain C and separate style domains S1 and S2. This model is learned in two stages – an unpaired pre-training stage and a paired fine-tuning stage, both requiring only image data. (b) The reconstruction stage applies the content/style model as a content consistency operator (bottom) within an ISTA-based iterative scheme. Given an aligned reference image, guidance is introduced into the reconstruction by simply replacing its aliased content with content derived from the reference. The 'refine' block denotes a content refinement update, which iteratively corrects for inconsistencies between the reference content and the measured k-space data, improving the effectiveness of the content consistency operator. DC denotes data consistency. For comparison, two other reconstruction priors are shown, namely wavelet-domain soft-thresholding (top) and CNN-based denoising (middle) used in CS and PnP-CNN algorithms, respectively.">
</p>

[![MedIA](https://img.shields.io/badge/DOI-10.1016/j.media.2026.104160-blue)](https://www.sciencedirect.com/science/article/pii/S136184152600229X)
[![arXiv](https://img.shields.io/badge/arXiv-2409.13477---?logo=arXiv&labelColor=b31b1b&color=grey)](https://arxiv.org/abs/2409.13477)

Reference implementation of the PnP-CoSMo algorithm for multi-contrast MRI reconstruction as proposed in:
- Rao, et al. *"A Plug-and-Play Method for Guided Multi-contrast MRI Reconstruction based on Content/Style Modeling"* [(Medical Image Analysis, 2026)](https://www.sciencedirect.com/science/article/pii/S136184152600229X)
- Rao, et al. *"Guided Multicontrast Reconstruction based on the Decomposition of Content and Style."* [(ISMRM 2024, Oral)](https://archive.ismrm.org/2024/0808_0WSPOmQ8g.html)

Additionally, PnP-CoSMo was applied in the following work:
- Rao, et al. *"Analysis of the Information Contribution of Different Contrast Scans in an MRI Examination aided by Content/Style Modeling"* [(ISMRM 2026, Oral + AMPC Poster)]()
- Jabarimani, Rao, et al. *"Accelerated FLAIR Imaging at 0.6T using T2W-guided Multi-contrast Deep Learning-based Reconstruction using a Zero-shot Approach."* [(ISMRM 2025, Poster)](https://archive.ismrm.org/2025/2766.html)



## Code Overview
Directory structure:
```
pnp_cosmo
  |
  |- configs: Config files for content/style modeling
  |- cosmo: Content/style modeling code
  |- recon: Reconstruction algorithms and demo notebook
  |- data: Dataloaders
```



## Citation

This research was conducted at Leiden University Medical Center in collaboration with Philips as part of the [AI4MRI ICAI Lab](https://icai.ai/lab/ai4mri-lab/). The software contained here is for research purposes only and not for clinical use.

If you find this code to be useful in your research, please cite our paper:
```
@article{RAO2026104160,
  title = {A plug-and-play method for guided multi-contrast {MRI} reconstruction based on content/style modeling},
  journal = {Medical Image Analysis},
  pages = {104160},
  year = {2026},
  issn = {1361-8415},
  doi = {https://doi.org/10.1016/j.media.2026.104160},
  url = {https://www.sciencedirect.com/science/article/pii/S136184152600229X},
  author = {Chinmay Rao and Matthias {van Osch} and Nicola Pezzotti and Jeroen {de Bresser} and Mark {van Buchem} and Laurens Beljaards and Jakob Meineke and Elwin {de Weerdt} and Huangling Lu and Mariya Doneva and Marius Staring},
  keywords = {MRI reconstruction, Multi-modal, Content/style decomposition, Plug-and-play}
}
```
