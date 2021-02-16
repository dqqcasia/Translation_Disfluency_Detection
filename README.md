# Translation_Disfluency_Detection

We released 3,000 samples of the annotated spoken sentences and our annotation rules. 


This is a tool for disfluency detection in the pipeline speech translation system. 

## Requirements
* python==3.x 
* all packages in requirements are requested.
```
pip install -r requirements.txt
```

## Usage
1. Train a multi-task model for Chinese disfluency detection.
```
python train.py -c configs/test.yaml
```

