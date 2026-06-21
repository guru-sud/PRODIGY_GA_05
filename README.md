# PRODIGY_GA_05 — Neural Style Transfer

## Task
Apply the artistic style of one image (e.g., a famous painting) to the content of another image using neural style transfer.

## Overview
This project implements **Neural Style Transfer** two ways:

1. A **fast baseline** using a pre-trained TF-Hub arbitrary image stylization model
2. A **from-scratch implementation** using a pre-trained **VGG19** network as a fixed feature extractor, following the original Gatys et al. (2015) approach

## How It Works
- VGG19, pre-trained on ImageNet, is used purely as a feature extractor — its weights are never updated.
- **Content representation:** extracted from layer `block5_conv2`, capturing high-level structure/objects.
- **Style representation:** extracted from five layers (`block1_conv1` through `block5_conv1`), capturing texture and color patterns at multiple scales.
- Style is encoded using the **Gram matrix** of feature maps, which measures correlations between feature channels rather than spatial layout — this is what makes the result reflect "style" rather than just copying the painting's content.
- The **output image itself** (not the network) is optimized via gradient descent to minimize a combined content + style loss.
- **Total variation loss** is added to reduce high-frequency noise and produce a smoother, more visually coherent result.

## Images Used
- **Content image:** A photo of a sea turtle
- **Style image:** Kandinsky's *Composition VII* (can be swapped for any famous painting, e.g. Van Gogh's *Starry Night*)

## How to Run
1. Open `PRODIGY_GA_05.ipynb` in Google Colab
2. Run all cells in order: **Runtime → Run all**
3. GPU is optional but speeds things up — total runtime is a few minutes either way

## Dependencies
```
tensorflow
tensorflow_hub
numpy
matplotlib
Pillow
```

## Results
- Quick stylized result from the pre-trained TF-Hub model (baseline)
- From-scratch VGG19-based stylization showing the optimization process epoch by epoch
- Final smoothed result using total variation loss
- Side-by-side comparison of content, style, and final stylized output

## Reference
- [How Do Neural Style Transfers Work? — Towards Data Science](https://towardsdatascience.com/how-do-neural-style-transfers-work-b76de101eb3)
- [Neural Style Transfer with TensorFlow — GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/neural-style-transfer-with-tensorflow/)
- [Neural Style Transfer — TensorFlow Core Tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)
- Gatys, Ecker, Bethge, *A Neural Algorithm of Artistic Style* (2015)

## Author
guru-sud — Prodigy InfoTech Generative AI Internship, Task-05
