# Armol
This is the code for paper "Cost Effective MLaaS Federation: A Combinatorial Reinforcement Learning Approach" published in INFOCOM 2022.
## Setup
```bash
conda create --name armol --file requirements.txt
conda activate armol
```
## Running Armol for pre-processed real-trace data
Download preprocessed prediction [data]() and decompress it in `data` directory.
```
python train.py
```
## Dataset description
** The link of data will be established soon. **
 We request AWS Rekognition (AWS), Azure Computer Vision (AZU), Google Cloud Vision AI (GCP), and Aliyun Object Detection Service (ALI) 
 via Python SDK and capture the TCP packets by `tcpdump`. 
### Prediction
We request AWS, AZU, GCP, and ALI with COCO Validation Set (~4952 images). 

Download: [raw data]() | [preprocessed data]().

### Latency
We only record the packet files from AWS and AZU.
You can parse the latency by indexing the 

## Data sample
### Prediction
**AWS**
**AZU**
**GCP**
**ALI**
### Latency
**AWS**
**AZU**

## Citation
If you use any part of this code in your research, please cite our paper:
```
@inproceedings{xie2022cost,
  title={Cost Effective MLaaS Federation: A Combinatorial Reinforcement Learning Approach},
  author={Xie, Shuzhao and Xue, Yuan and Zhu, Yifei and Wang, Zhi},
  booktitle={IEEE INFOCOM 2022-IEEE Conference on Computer Communications},
  pages={1--10},
  year={2022},
  organization={IEEE}
}
```

## Acknowledgment
Shuzhao Xie thanks Chen Tang, Jiahui Ye, Shiji Zhou, and Wenwu Zhu for their help in making this work possible.
Yifei Zhu's work is funded by the SJTU Explore-X grant. 
This work is supported in part by NSFC (Grant No. 61872215), and Shenzhen Science and Technology Program (Grant No. RCYX20200714114523079). 
We would like to thank Tencent for sponsoring the research.
