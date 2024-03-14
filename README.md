# GloVe-Pytorch-file

### about

The original code is from https://github.com/HIT-SCIR/plm-nlp-code/tree/main/chp5

This is a training code, modified to handle large datasets:

- data is loaded using generators
- text data is in gensim PathLineSentences format (a directory with txt/gz files with one sentence per line, tokens separated by spaces)
- in addition to in-memory, added the option to write co-occurence matrix to disk using LMDB (very slow, needs optimization)

### usage:

edit parameters after the imports block, then

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python glove-pytorch-train.py
```

### environment setup:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install conda-forge::gensim -y
conda install conda-forge::lmdb -y
conda install conda-forge::tqdm -y

```

### to do:

- add muli-GPU support
- evaluation code
- dataset preparation code
