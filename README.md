# Armol [[arXiv]](https://arxiv.org/pdf/2204.13971.pdf) [[Slides]](./assets/armol-slides.pdf) [[Project]](https://mmlabsigs.notion.site/Cost-Effective-MLaaS-Federation-A-Combinatorial-Reinforcement-Learning-Approach-fd27b02a403240f3b55ec26ad5ba00db) 
This is the code for paper "Cost Effective MLaaS Federation: A Combinatorial Reinforcement Learning Approach" published in INFOCOM 2022.
## Setup
### Environment
We recommend to setup the environment with anaconda.
```bash
conda create -n mfed python=3.6
conda activate mfed
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
pip install -r requirements.txt
```
### Prepare prediction dataset
1. Build `data` and `results` directory;
    ```bash
    mkdir data
    mkdir results
    ```

2. Download [preprocessed prediction dataset](https://drive.google.com/file/d/1DyxMAtetl6RlLJUVEM_mC8_C8MnFMUog/view?usp=sharing), which contains `train` and `test` directory, extract them into the `data` directory. There are `aws`, `azure`, `google`, and `ground-truth` dirs in each dir. The `ground-truth` is derived from the [COCO dataset](https://cocodataset.org/#download), whose origin form is a json file for all images. I transform the origin json file into a directory contains a series of texts files for the purpose of training RL models. The `train/test/*/0.txt` stands for the ground truth or predictions of train/test image. The mapping of image's name to its rank can be found at `src/scripts/*rank.json`.
   
3. Download the [COCO dataset 2017 Train and Val Images](https://cocodataset.org/#download), then extract these images into `data/train/images` and `data/test/images` directories, respectively. Due to the ground truth of COCO 2017 test images has not been released, we put the validation set as the test set in our experiments.
   
4. Replace the `WORKDIR` in `src/common.py` with your current work directory.

## Run Armol
Just enter the command below to train the RL model. You can tune the hyper-parameters in `train.py`. I think this paradigm works the same as `argparser`, so I don't want to write an args' parser.
```
python train.py
```
## Latency dataset description
### Setup
1. We use `scapy` to parse `.pcap` files, so first install it.
    ```bash
    pip install scapy
    ```
2. Download latency records that in `.pcap` format. Extract them in anywhere you want. Just modify the input data paths (maybe `lag_dir` or other variables) to the path you store the `.pcap` files. These paths are in latency analysis files that placed in `src/latency` directory. Because these files require a huge place to store, I upload them to Baidu NetDisk. Here are the link and key:
    ```bash
    link: https://pan.baidu.com/s/15xW6xPZ5ubVJj_WmZi-HLg
    key: stv8 
    ```

### Analyze latency
We request AWS Rekognition (AWS), Azure Computer Vision (AZU), Google Cloud Vision AI (GCP), and Aliyun Object Detection Service (ALI) via Python SDK and capture the TCP packets by `tcpdump`. We devide the latency into transmission 
latency and inference latency. By indexing the packet with special HTTP code,
we distinguish the first packet and the last packet in a request. Thus we can get the full latency. And the transimission latency can be calculated by indexing the first packet and the last ack packet from server. We also find that the ip of MLaaS is not constant, and GCP would help the customer to locate the service with lowest latency. You can find the measurement time of a file by its name, such as `20210508_010101.cap`. And `us2sg` means send a request from a VM in the United States to Singapore. Please refer to the `src/latency` directory for more details. 

### How to measure the latency?
We record the packet files from AWS, AZU, and GCP with `tcpdump`. You can open these `.pcap` files with wireshark to get a better illustration. What's more, as we test the services in different data regions every hour, we leverage `crontab` to repeat these measurements for convenience. We send the requests according to the ranks in `src/deploy/event.json`. We only employ the COCO Validation Set to measure the latency for the sake of saving money. For more details, please refer to the `src/deploy` directory. 

If you want to repeat the measurments, please modify the `SUB_KEY` and `END_POINT` variables in `src/deploy/azure_sender.py` and config your aws and google cloud accounts. Please refer to their offical documentations for more details. 

## Raw data
If you want the raw returned predictions from AWS, AZU, and GCP, please send an email to `xiesz1999@163.com`. To be honest, I'm too lazy to sort through the clutter files if no one needs them. However, if you really need these files, feel free to ask me :).

## Synonyms table
This table is placed in `src/sripts/word2num.json`, most of the entries are refered from [WordNet](https://wordnet.princeton.edu).

## Details of simulated MLaaSes 
I borrowed the pre-trained models from [detectron2](https://github.com/facebookresearch/detectron2) and [Tensorflow model garden](https://github.com/tensorflow/models) to simulate 7 MLaaSes. If you want to replicate the expertiment, please transform the predictions of these models into the format that utilized in `data/train/google`. For more details, please refer to `src/simulate`.

```python
MODELS_SELECT = {
  0 : 'centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8', # 29.3
  1 : 'centernet_resnet50_v2_512x512_kpts_coco17_tpu-8', # 27.6
  2 : 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', # 22.2
  3 : 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8', # 28.2
  4 : 'ssd_mobilenet_v2_320x320_coco17_tpu-8', # 20.2
  5 : 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8', # 29.3
  6 : 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8' # 29.1
}
```

## Citation
If you use any part of our code or dataset in your research, please cite our paper:
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

## Appendix
There are also many other interesting discoveries, I will summary them when I get a day off. Also, you can find them by yourself. 
