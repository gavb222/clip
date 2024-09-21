# clip
This repo trains a pair of models based on Contrastive Language-Image Pretraining (CLIP) (https://arxiv.org/abs/2103.00020).
I am using the google image-caption pairs dataset (https://github.com/google-research-datasets/conceptual-captions) to train this model, and since it is very large, I am streaming the dataset from the internet as opposed to saving it on my machine.
