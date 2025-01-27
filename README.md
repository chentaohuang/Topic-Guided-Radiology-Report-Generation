# TGRG

The code of the paper:  Report is a Mixture of Topics: Topic-Guided Radiology Report
 Generation

### The code will be continuously optimized to enhance both simplicity and convenience in deployment.

## Step 1: Pre-process

Run `python preprocess.py` to construct topics and their relevant knowledge.

Since the process is tedious and time-consuming, we have uploaded the constructed knowledge for MIMIC-CXR to facilitate quick reproduction of the experiment for those interested.


## Step 2: Train/Evaluation

```
python main.py 
```

# Acknowledgment:
The code implementation is modified from the project: https://github.com/zhjohnchan/R2Gen

The code for extracting topic n-grams is adapted from the project: https://github.com/wjhou/ORGan
