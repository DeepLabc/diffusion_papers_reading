


Any-Size-Diffusion: Toward Efficient Text-Driven Synthesis for Any-Size HD Images
Abstract
Stable diffusion, a generative model used in text-to-image synthesis, frequently encounters resolution-induced composition problems when generating images of varying sizes. This issue primarily stems from the model being trained on
pairs of single-scale images and their corresponding text descriptions. Moreover, direct training on images of unlimited sizes is unfeasible, as it would require an immense number of text-image pairs and entail substantial computational expenses. To overcome these challenges, we propose a two-
stage pipeline named Any-Size-Diffusion (ASD), designed to efficiently generate well-composed images of any size, while minimizing the need for high-memory GPU resources. Specifically, the initial stage, dubbed Any Ratio Adaptability
Diffusion (ARAD), leverages a selected set of images with a restricted range of ratios to optimize the text-conditional diffusion model, thereby improving its ability to adjust composition to accommodate diverse image sizes. To support the
creation of images at any desired size, we further introduce a technique called Fast Seamless Tiled Diffusion (FSTD) at the subsequent stage. This method allows for the rapid enlargement of the ASD output to any high-resolution size, avoiding seaming artifacts or memory overloads. Experimental
results on the LAION-COCO and MM-CelebA-HQ benchmarks demonstrate that ASD can produce well-structured images of arbitrary sizes, cutting down the inference time by 2× compared to the traditional tiled algorithm.

![image](https://github.com/DeepLabc/diffusion_papers_reading/assets/43690274/39727b55-a604-417a-a04b-005489c29e91)
