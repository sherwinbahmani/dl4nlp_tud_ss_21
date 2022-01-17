# Deep Learning for Natural Language Processing SS 2021 (TU Darmstadt)

## Task
Training huge unsupervised deep neural networks yields to strong progress in the field of Natural Language Processing (NLP). Using these extensively pre-trained networks for particular NLP applications is the current state-of-the-art approach. In this project, we approach the task of ranking possible clarifying questions for a given query. We fine-tuned a pre-trained BERT model to rank the possible clarifying questions in a classification manner. The achieved model scores a top-5 accuracy of 0.4565 on the provided benchmark dataset.

## Installation
This project was originally developed with Python 3.8, PyTorch 1.7, and CUDA 11.0. The training requires 
one NVIDIA GeForce RTX 1080 (11GB memory).

- Create conda environment:
```
conda create --name dl4nlp
source activate dl4nlp
```
- Install the dependencies:
```
pip install -r requirements.txt
```

## Run
We use a pretrained [BERT-Base](https://arxiv.org/abs/1810.04805) by [Hugging Face](https://huggingface.co/docs/transformers/model_doc/bert#bertmodel) and fine-tune it on the given training dataset.
To run training, please use the following command:

```
python main.py --train
```

For evaluation on the test set, please use the following command:
```
python main.py --test
```

Arguments for training and/or testing:
- ```--train```: Run training on training dataset. Default: `True`
- ```--val```: Run evaluation during training on validation dataset. Default: `True`
- ```--test```: Run evaluation on test dataset. Default: `True`
- ```--cuda-devices```: Set GPU index Default: `0`
- ```--cpu```: Run everything on CPU. Default: `False`
- ```--data-parallel```: Use DataParallel. Default: `False`
- ```--data-root```: Path to dataset folder. Default: `data`
- ```--train-file-name```: Name of training file name in data-root. Default: `training.tsv`
- ```--test-file-name```: Name of test file name in data-root. Default: `test_set.tsv`
- ```--question-bank-name```: Name of question bank file name in data-root. Default: `question_bank.tsv`
- ```--checkpoints-root```: Path to checkpoints folder. Default: `checkpoints`
- ```--checkpoint-name```: File name of checkpoint checkpoints-root to start training or use for testing. Default: `None`
- ```--runs-root```: Path to output runs folder for tensorboard. Default: `runs`
- ```--txt-root```: Path to output txt folder for evaluation results. Default: `txt`
- ```--lr```: Learning rate. Default: `1e-5`
- ```--betas```: Betas for optimization. Default: `(0.9, 0.999)`
- ```--weight-decay```: Weight decay. Default: `1e-2`
- ```--val-start```: Set at which epoch to start validation. Default: `0`
- ```--val-step```: Set at which epoch rate to valide. Default: `1`
- ```--val-split```: Use subset of training dataset for validation. Default: `0.005`
- ```--num-epochs```: Number of epochs for training. Default: `10`
- ```--batch-size```: Samples per batch. Default: `32`
- ```--num-workers```: Number of workers. Default: `4`
- ```--top-k-accuracy```: Evaluation metric with flexible top-k-accuracy. Default: `50`
- ```--true-label```: True label in dataset. Default: `1`
- ```--false-label```: False label in dataset. Default: `0`

### Example output
```User query```:

Tell me about Computers

```Propagated clarifying questions```:

1) do you like using computers
2) do you want to know how to do computer programming
3) do you want to see some closeup of a turbine
4) are you looking for information on different computer programming languages
5) are you referring to a software
