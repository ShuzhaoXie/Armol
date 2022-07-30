import os
import time
import urllib

import cv2 as cv
import gym
import numpy as np
# from sklearn.metrics import mean_squared_error, r2_score
import timm
import torch
# from contextualmab.mobilenetv3 import MobileNetV3_Small
from gym.utils import seeding
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from src.ablation import wbf
from src.common import (ALPHA, CUR_DEVICE, ONLINE, TRAIN_IMAGE_DIR, TRAIN_RANK, Detections,
                    file_lines_to_list, to_numpy)
from src.reward_generator import compare_gt_m2

model = timm.create_model('mobilenetv3_large_100', pretrained=True)
model.to(CUR_DEVICE)
model.eval()
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

def cal_l1_distance(a, b):
    if len(a) != len(b):
        return 100.0
    ret = 0
    for i, ai in enumerate(a):
        ret += np.square(ai - b[i])
    return ret * 1.0


class ARMOR_TRAIN(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ARMOR_TRAIN, self).__init__()
        self.action_space = gym.spaces.Box(np.array([0] * 3), np.array([1] * 3), dtype=np.float16)
        self.observation_space = gym.spaces.Box(np.array([0] * 1000), np.array([1] * 1000), dtype=np.float32)
        self.t = 0
        self.gt = Detections('gt_train', TRAIN_RANK)
        self.aws = Detections('aws_train', TRAIN_RANK)
        self.azure = Detections('azure_train', TRAIN_RANK)
        self.google = Detections('google_train', TRAIN_RANK)
        # self.best_actions = parse_pattern(WORK_DIR + '/dds/src/upper_bound_{}_pattern.json'.format(box_fusion_mode))
        self.image_rank = TRAIN_RANK
        self.image_path = os.path.join(TRAIN_IMAGE_DIR, self.image_rank[self.t])
        self.nms_mode = wbf
        print('load best actions')

    def step(self, action):
        action = to_numpy(action, dtype=np.int32)
        reward = self._get_reward(action)
        if self.t == 19999:
            done = True
        else:
            done = False
        self.t += 1
        if self.t <= 19999:
            self.image_path = os.path.join(TRAIN_IMAGE_DIR, self.image_rank[self.t])
            state = self._get_next_state()
        else:
            state = torch.zeros(1000, device=CUR_DEVICE)
        info = {'State': state, 'Reward': reward}
        if done:
            print('Trajecty Done')
        return state, reward, done, info

    def get_results(self, act):
        res = []
        
        if act[0] == 1:
            res.extend(self.aws.get(self.t))
        if act[1] == 1:
            res.extend(self.azure.get(self.t))
        if act[2] == 1:
            res.extend(self.google.get(self.t))

        res = self.nms_mode(res)

        return res

    def _get_reward(self, act):
        if ONLINE:
            # print('online')
            infer = self.get_results(act)
            whole = self.get_results([1, 1, 1])

            if len(infer) == 0 or len(whole) == 0:
                r1 = -1
            else:
                r1 = compare_gt_m2(infer, whole)
            # print('myenv.py infer_lags', infer_lags)
            # cost = sum(act)

            reward = np.tanh(r1) - 0.15
        else:
            infer = self.get_results(act)

            # for anno in infer:
            #     print(anno.to_str())

            cur_gt = self.gt.get(self.t)
            # for anno in cur_gt:
            #     print(anno.to_str())
            if len(infer) == 0:
                r1 = -1
            else:
                r1 = compare_gt_m2(infer, cur_gt)

            s_act = np.sum(act)
            
            # print('r_1 act', r1, s_act)
            # mix reward
            r1 = r1 + s_act * ALPHA

            reward = np.tanh(r1) - 0.15

            # print(reward)

        return reward

    # def _get_next_state(self):
    #     image_name = self.image_rank[self.t]
    #     img = Image.open(os.path.join(IMAGE_DIR_PATH, image_name)).convert('RGB')
    #     tensor = transform(img).unsqueeze(0).to(CUR_DEVICE)
    #     with torch.no_grad():
    #         out = model(tensor)
    #     probabilities = torch.nn.functional.softmax(out[0], dim=0)
    #     top5_prob, top5_catid = torch.topk(probabilities, 20)
    #     out = np.zeros(1000)
    #     for i in range(top5_catid.size(0)):
    #         out[top5_catid[i].item()] = 1
    #     return to_tensor(out)

    def _get_next_state(self):
        image_name = self.image_rank[self.t]
        img = Image.open(os.path.join(TRAIN_IMAGE_DIR, image_name)).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(CUR_DEVICE)
        with torch.no_grad():
            out = model(tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        # print(type(probabilities), probabilities.shape)
        return probabilities

    def reset(self):
        self.t = 0
        state = self._get_next_state()
        self.image_path = os.path.join(TRAIN_IMAGE_DIR, self.image_rank[self.t])
        return state

    def render(self, mode='human'):
        print(f'Step: {self.t}')
