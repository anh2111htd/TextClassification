Text Classification
=======================
This project includes state-of-the-art implementations of text and text pair classification.

## Quick start

### Train and eval CNN on classifying synthetic sequence pair
```
python main.py
```

### Experiment visualization with Tensorboard
```
bash run_tb.sh
```

## CLI Usage
```
python main.py  [--task -t <task-name>]
                [--model -m <model-name>]
parameters:
    --task -t           default: "synthetic"    synthetic|stance
    --model -m          default: "cnn"          cnn|bidaf
```

## Tested environment
```
python==3.6.6
spacy==2.1.4
en-core-web-sm==2.1.0
numpy==1.16.4
pandas==0.25.0
scikit-learn==0.21.2
tensorboard==1.14.0
torch==1.2.0
torchtext==0.4.0
```
