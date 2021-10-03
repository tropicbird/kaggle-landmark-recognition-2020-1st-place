# Kaggle Landmark Recognition 2020 competition: Winner solution

This repository contains the code for our winning solution to the 2020 edition of the Google Landmark Recognition competition hosted on Kaggle: <br />
https://www.kaggle.com/c/landmark-recognition-2020/leaderboard

The full solution is described in a paper hosted on arxiv: <br /> 
https://arxiv.org/abs/2010.01650

In order to run this code you need the train and test data from GLDv2: <br />
https://github.com/cvdfoundation/google-landmark

To train a model, please run ```src/train.py``` with a config file as flag:
```
python train.py --config config1
```

You need to adjust data paths and other parameters in respective config file to make it work.

The blending and ranking procedure is detailed in ```notebooks/blend_ranking.ipynb```.

# [Modified the coriginal code for personal use] Last update 2021/10/03

For Kaggle Landmark Recognition 2021 competition, I modified the original code created by [psinger](https://github.com/psinger). I basically utilized the config1 (backbone: gluon_seresnext101_32x4d) and config7 (backbone: res2net101_26w_4s) to create my submission in the 2021 version of the competition.
