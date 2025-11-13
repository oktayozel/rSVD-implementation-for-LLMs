### Abstract

Multi Head Latent Attention Uses SVD for matrix compression. Our goal it to implement randomized-SVD instead of SVD, and make the process more efficient.

### Standard SVD Implementation

[Standard SVD README](https://github.com/cangokmen/CS599-Randomized-SVD/blob/master/docs/README(SVD).md)

### How to run?
There are multiple versions of the GPT implementation in this repository. 

In any case you are required to install the dependencies:

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

and for testing purposes you can prepare a small dataset as follows:

```sh
python train.py config/train_shakespeare_char.py
```

In order to run regular attention version:


In order to run Multi-Head Latent Attention with Regular SVD version:


In order to run Multi-Head Latent Attention with Randomized SVD version:






### acknowledgements
MLA implementation is based on nanoGPT implementation of Karpathy.
https://github.com/karpathy/nanoGPT
