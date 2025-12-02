# ViT-Notes
This is a repository for recoding the knowledge during Vision Transformer learning.

## Prerequisites
### Patch Embedding
First, we need to divide the image into patches and project these regions onto the feature space for obtain the image tokens which shape is `(B, N, C)`, indicate the `Batch`、 `Number of Patches` and `Channel of feature dimension` respectively.

Second, unlike Natural Language Processing (NLP) tokens as a continue sequence, we split our complete image matrix into seperate part. Therefore, **Positional Encoding** is essential for supporting the positional dependency between each patches. 


### Self-Attention Mechanism



### Weight Initialization
By default, `nn.Linear(in_features, out_features)` uses a small-variance initialization, which helps stabilize early training on ***large*** datasets.

For ***smaller*** or ***more difficult*** datasets, using a larger initialization variance can improve performance by increasing the diversity of early weight representations.



## Update Log
`models/transformer.py` references [[1]](#reference) to reproduce the architecture.

`models/vit_mnist.py` references [[2]](#reference) and [[3]](#reference) to reproduce and train a ViT model on MNIST dataset.<br>
<sub>*Modify the architecture based on [[3]](#reference) to accelerate the efficiency and maintain the accuracy*</sub>


## Reference
### 📕Paper
[**Transformer**](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf): *Attention Is All You Need* (**2017 NeurIPS**) 

[**ViT**](https://arxiv.org/pdf/2010.11929/1000): *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* (**2021 ICLR**) 

[**TimeSformer**](https://proceedings.mlr.press/v139/bertasius21a/bertasius21a.pdf): *Is Space-Time Attention All You Need for Video Understanding?* (**2021 ICML**) 

### 🔗NoteBook
[[1]](https://nlp.seas.harvard.edu/annotated-transformer/) The Annotated Transformer [[2]](https://medium.com/analytics-vidhya/illustrated-vision-transformers-165f4d0c3dd1)  Illustrated Vision Transformers 

[[3]](https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c) Vision Transformers from Scratch (Pytorch) 
