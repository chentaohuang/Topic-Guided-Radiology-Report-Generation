# TGRG

The code of the paper:  Report is a mixture of topic: Topic-Guided Radiology Reporte
 Generation

### The code would be formulated to enhance both simplicity and convenience in deployment, if time is available.

## Step 1: Pre-process

Run `python preprocess.py` to construct topics and their relevant knowledge.

Since the process is tedious and time-consuming, we have uploaded the constructed knowledge for MIMIC-CXR to facilitate quick reproduction of the experiment for those interested:https://drive.google.com/drive/folders/1dg0-Q_uciRrkB0-ok18moOOUch1O05bP?usp=drive_link


## Step 2: Train/Evaluation

```
python main.py 
```

# Acknowledgement:
The code implementation is modified from the project: https://github.com/zhjohnchan/R2Gen

The code for extracting topic n-grams is adapted from the project: https://github.com/wjhou/ORGan
