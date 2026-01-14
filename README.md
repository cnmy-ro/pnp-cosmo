# PnP-CoSMo

Reference implementation of the `PnP-CoSMo` algorithm for multi-contrast MRI reconstruction as proposed in:
- Rao, van Osch, Pezzotti, de Bresser, van Buchem, Beljaards, Meineke, de Weerdt, Lu, Doneva, Staring. *"A Plug-and-Play Method for Guided Multi-contrast MRI Reconstruction based on Content/Style Modeling"* [(arXiv 2025)](https://arxiv.org/abs/2409.13477)
- Rao, Beljaards, van Osch, Doneva, Meineke, Schülke, Pezzotti, de Weerdt, Staring. *"Guided Multicontrast Reconstruction based on the Decomposition of Content and Style."* [(ISMRM 2024)](https://cnmy-ro.github.io/section-research/)

![Overview of PnP-CoSMo, our two-stage approach to guided reconstruction. (a) The first stage learns a content/style model of two-contrast MR image data offline. We assume that the two image domains X1 and X2 can be decomposed into a shared content domain C and separate style domains S1 and S2. This model is learned in two stages – an unpaired pre-training stage and a paired fine-tuning stage, both requiring only image data. (b) The reconstruction stage applies the content/style model as a content consistency operator (bottom) within an ISTA-based iterative scheme. Given an aligned reference image, guidance is introduced into the reconstruction by simply replacing its aliased content with content derived from the reference. The "refine" block denotes a content refinement update, which iteratively corrects for inconsistencies between the reference content and the measured k-space data, improving the effectiveness of the content consistency operator. DC denotes data consistency. For comparison, two other reconstruction priors are shown, namely wavelet-domain soft-thresholding (top) and CNN-based denoising (middle) used in CS and PnP-CNN algorithms, respectively.](docs/GraphicalAbstract.png)

Additionally, `PnP-CoSMo` was used in the following work:
- Jabarimani, Rao, Ercan, Dong, Pezzotti, Doneva, Weerdt, van Osch, Staring, Nagtegaal. *"Accelerated FLAIR Imaging at 0.6T using T2W-guided Multi-contrast Deep Learning-based Reconstruction using a Zero-shot Approach."*[(ISMRM 2025)](https://archive.ismrm.org/2024/0808_0WSPOmQ8g.html)


---
This research was done at LUMC in collaboration with Philips as part of the [AI4MRI ICAI Lab](https://icai.ai/lab/ai4mri-lab/). This software is for research purposes only and not for clinical use.

If you find this code to be useful in your research, we would appreciate if you cite our paper:
```
@article{rao2024plug,
  title={A Plug-and-Play Method for Guided Multi-contrast MRI Reconstruction based on Content/Style Modeling},
  author={Rao, Chinmay and van Osch, Matthias and Pezzotti, Nicola and de Bresser, Jeroen and van Buchem, Mark and Beljaards, Laurens and Meineke, Jakob and de Weerdt, Elwin and Lu, Huangling and Doneva, Mariya and others},
  journal={arXiv preprint arXiv:2409.13477},
  year={2024}
}
```