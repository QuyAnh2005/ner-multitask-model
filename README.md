# Multi-task Model with NER and POS

## Pre-requisites
- Install python requirements. Please refer [requirements.txt](requirements.txt)
```bash
pip install -r requirements.txt
```

- Preprocessing data
```python
python preprocessing.py -c configs/base.json -m pos-ner
```

## Training Example
```python
python train.py -c configs/base.json -m pos-ner
```
