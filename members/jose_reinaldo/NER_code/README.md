# Supervised training of NER models

Code for training of deep learning models applied to named entity recognition tasks.

## Models
- CNN-CNN-LSTM (proposed by [Shen et al](https://arxiv.org/abs/1707.05928))
- CNN-biLSTM-CRF (proposed by [Ma and Hovy](https://www.aclweb.org/anthology/P16-1101/))

## Arguments available for the Supervised_new.py script
---

- --save_training_path : indicates path to save training history of the model

- --save_model_path : Indicates path to save trained model


### dataset parameters
- --train_path : Path to load training set from

- --test_path : Path to load testing set from

- --dataset_format : Format of the dataset (e.g. iob1, iob2, iobes)

### Embedding parameters
- --embedding_path : Path to load pretrained embeddings from

- --augment_pretrained_embedding : Indicates whether to augment pretrained embeddings with vocab from training set')

### General model parameters
- --model : Neural NER model architecture

- --char_embedding_dim : Embedding dimension for each character

- --char_out_channels : # of channels to be used in 1-d convolutions to form character level word embeddings

### CNN-CNN-LSTM specific parameters
- --word_out_channels : # of channels to be used in 1-d convolutions to encode word-level features

- --word_conv_layers : # of convolution blocks to be used to encode word-level features

- --decoder_layers : # of layers of the LSTM greedy decoder

- --decoder_hidden_size : Size of the LSTM greedy decoder layer

### CNN-biLSTM-CRF specific parameters
- --lstm_hidden_size : Size of the lstm for word-level feature encoder

### Trainign hyperparameters
- --lr : Learning rate for NER mdoel training

- --grad_clip : Value at which to clip the model gradient throughout training

- --momentum : Momentum for the SGD optimization process

### Training parameters
- --epochs : Number of supervised training epochs

- --batch_size : Batch size for training

