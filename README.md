


Any-Size-Diffusion: Toward Efficient Text-Driven Synthesis for Any-Size HD Images [paper](https://arxiv.org/abs/2308.16582)

Abstract

Stable diffusion, a generative model used in text-to-image synthesis, frequently encounters resolution-induced composition problems when generating images of varying sizes. This issue primarily stems from the model being trained on pairs of single-scale images and their corresponding text descriptions. Moreover, direct training on images of unlimited sizes is unfeasible, as it would require an immense number of text-image pairs and entail substantial computational expenses. To overcome these challenges, we propose a two-stage pipeline named Any-Size-Diffusion (ASD), designed to efficiently generate well-composed images of any size, while minimizing the need for high-memory GPU resources. Specifically, the initial stage, dubbed Any Ratio Adaptability
Diffusion (ARAD), leverages a selected set of images with a restricted range of ratios to optimize the text-conditional diffusion model, thereby improving its ability to adjust composition to accommodate diverse image sizes. To support the
creation of images at any desired size, we further introduce a technique called Fast Seamless Tiled Diffusion (FSTD) at the subsequent stage. This method allows for the rapid enlargement of the ASD output to any high-resolution size, avoiding seaming artifacts or memory overloads. Experimental
results on the LAION-COCO and MM-CelebA-HQ benchmarks demonstrate that ASD can produce well-structured images of arbitrary sizes, cutting down the inference time by 2× compared to the traditional tiled algorithm.

![image](https://github.com/DeepLabc/diffusion_papers_reading/assets/43690274/39727b55-a604-417a-a04b-005489c29e91)

高效率的任意大小图像生成方法，思路：  

（1）：使用不同比例的图像进行训练，常见的训练一般是256x256或者512x512大小的图像，作者在训练时引入一组预定义的比例因子r：s,一个r对于一个s,s表示训练时的图像大小，然后判断每张图像的H/W与哪个r最近，进而resize到对于大小的尺寸进行训练  

（2）：推理阶段先生成预设大小的图片，然后以该图像为condition输入到另一个diffusion模型中，获得任意大小的图像  

---

Adding Conditional Control to Text-to-Image Diffusion Models [paper](https://arxiv.org/abs/2302.05543) [code](https://github.com/lllyasviel/ControlNet)

Abstract

We present ControlNet, a neural network architecture to
add spatial conditioning controls to large, pretrained text-
to-image diffusion models. ControlNet locks the production-
ready large diffusion models, and reuses their deep and ro-
bust encoding layers pretrained with billions of images as a
strong backbone to learn a diverse set of conditional controls.
The neural architecture is connected with “zero convolutions”
(zero-initialized convolution layers) that progressively grow
the parameters from zero and ensure that no harmful noise
could affect the finetuning. We test various conditioning con-
trols, e.g., edges, depth, segmentation, human pose, etc., with
Stable Diffusion, using single or multiple conditions, with
or without prompts. We show that the training of Control-
Nets is robust with small (<50k) and large (>1m) datasets.
Extensive results show that ControlNet may facilitate wider
applications to control image diffusion models.

![image](https://github.com/DeepLabc/diffusion_papers_reading/assets/43690274/989a32e3-bd70-4379-b851-04743f39c9d6)


![image](https://github.com/DeepLabc/diffusion_papers_reading/assets/43690274/7bd8410a-540f-4dac-983f-15ace05ff38a)


通过其他condition控制diffusion的输出，思路：

训练时冻结住stable diffusion的权重，单独增加一个网络（ControlNet）接受condition的输入，在ControlNeT中，每个encoder block的权重直接使用stable diffusion的权重（保留encoder强大的学习能力），controlnet与stable diffusion的每一层decoder连接时，通过一个zero初始化的卷积层作为桥梁连接，这样在训练开始时，有害噪声就不会影响trainable copy中神经网络层的隐藏状态

---
GLIGEN: Open-Set Grounded Text-to-Image Generation [paper](https://arxiv.org/abs/2301.07093) [code](https://github.com/gligen/GLIGEN)

Abstract

Large-scale text-to-image diffusion models have made
amazing advances. However, the status quo is to use
text input alone, which can impede controllability. In this
work, we propose GLIGEN, Grounded-Language-to-Image
Generation, a novel approach that builds upon and extends
the functionality of existing pre-trained text-to-image diffusion models by enabling them to also be conditioned on
grounding inputs. To preserve the vast concept knowledge of
the pre-trained model, we freeze all of its weights and inject
the grounding information into new trainable layers via a
gated mechanism. Our model achieves open-world grounded
text2img generation with caption and bounding box condition inputs, and the grounding ability generalizes well to
novel spatial configurations and concepts. GLIGEN’s zero-shot performance on COCO and LVIS outperforms existing
supervised layout-to-image baselines by a large margin.



![image](https://github.com/DeepLabc/diffusion_papers_reading/assets/43690274/5cd6b450-d196-4718-8125-8f5eb277873f)

![image](https://github.com/DeepLabc/diffusion_papers_reading/assets/43690274/26be4eae-7f1c-4aad-9905-4d83190069e0)

stable diffusion的可控性拓展，思路：

原始的stable diffusion的结构不变，训练时也是直接使用原始权重，核心在于添加了一个gated self-attention，将grouding feature与中间的vision feature拼接之后做一个self attention，然后只取vision feature输入到下一个layer（还有一个可学习参数scale）, 这样的目的是赋予stable diffusion语言到图像生成模型新的空间接地能力

---










