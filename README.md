# GloVe-Pytorch-file

The original code is from https://github.com/HIT-SCIR/plm-nlp-code/tree/main/chp5

This is a training code, modified to handle large datasets:

- data is loaded using generators
- co-oocurence matrix is written to disk and loaded using memory mapping

To do:

- add muli-GPU support
- evaluation code
