# Active learning training of NER models

- Author: Jose Reinaldo Cunha Santos A V Silva Neto
- Filiation: University of Brasilia (2019-2021)

Code for training deep learning models applied to named entity recognition tasks using active learning and self-learning. Pure active learning simulation is also possible, see *Token-level self-labeling parameters* section below for more details

## Models
- CNN-CNN-LSTM (proposed by [Shen et al](https://arxiv.org/abs/1707.05928))
- CNN-biLSTM-CRF (proposed by [Ma and Hovy](https://www.aclweb.org/anthology/P16-1101/))

## Arguments available for the ActiveLearning.py script
---

- --save_training_path : indicates path to save training history of the model

- --save_model_path : Indicates path to save trained model


### dataset parameters
- --train_path : Path to load training set from

- --valid_path : Path to load validation (dev) set from

- --test_path : Path to load testing set from

- --dataset_format : Format of the dataset (e.g. iob1, iob2, iobes)

### Embedding parameters
- --embedding_path : Path to load pretrained embeddings from (only .kv files supported so far)

- --~~augment_pretrained_embedding : Indicates whether to augment pretrained embeddings with vocab from training set')~~

### General model parameters
- --model : Neural NER model architecture. Supported architectures are:
    - CNN-CNN-LSTM
    - CNN-biLSTM-CRF

- --char_embedding_dim : Embedding dimension for each character

- --char_out_channels : # of channels to be used in 1-d convolutions to form character level word embeddings

### CNN-CNN-LSTM specific parameters
- --word_out_channels : # of channels to be used in 1-d convolutions to encode word-level features

- --word_conv_layers : # of convolution blocks to be used to encode word-level features

- --decoder_layers : # of layers of the LSTM greedy decoder

- --decoder_hidden_size : Size of the LSTM greedy decoder layer

### CNN-biLSTM-CRF specific parameters
- --lstm_hidden_size : Size of the lstm for word-level feature encoder

### Trainign parameters
- --epochs : Maximum number of training epochs to  be used in a single iteration of the active learning algorithm

- --lr : Learning rate for the model training

- --grad_clip : Value at which to clip the model gradient throughout training

- --momentum : Momentum for the SGD optimization process

- --batch_size : Batch size for training

- --early_stop_method : Early stopping technique to be used
    - DevSetF1 : Uses f1-score on validation set
    - DevSetLoss : Uses mean loss on validation set
    - DUTE : Uses DUTE strategy proposed by [C.S.A.V.S. Neto and Faleiros](https://link.springer.com/chapter/10.1007/978-3-030-91699-2_28)
    - full_epochs : Trains model for the full number of training epochs without early stopping

- --patience : Number of training epochs without model improvement for early stopping to occurr

### Active learning parameters

- --initial_set_seed : Seed that controls the random sampling for the initial set of labeled samples. If same seed is used, the initial set of labeled samples is always the same.

- --labeled_percent_stop : Minimum percentage of the training set that must be in the labeled set for the active learning algorithm to stop. Value must be float in the interval [0,1]

- --query_fn : Sampling (querying) function to be used. Options implemented so far are:
    - random_sampling
    - least_confidence
    - normalized_least_confidence : Also called Maximum normalized log-probability (MNLP) and proposed by [Shen et al](https://arxiv.org/abs/1707.05928)

- --query_budget : Number of words that can be queried for annotation in a single iteration of the active learning algorithm

### Token-level self-labeling parameters

- --TokenSelfLabel_flag : Flag that indicates whether to perform token level self-labeling or not (0 to not use, 1 to use)

- --min_confidence : Minimum confidence the model needs to have in order to self-label a specific token
