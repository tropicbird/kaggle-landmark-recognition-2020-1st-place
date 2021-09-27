
import os
import albumentations as A
abs_path = os.path.dirname(__file__)

args = {
    'model_path':'../models/',
#     'data_path':'/ssd/kaggle_landmark/input/',
    'data_path':'../../input/', #Changed!!
#     'data_path_2019':'/ssd/kaggle_landmark/2019/',
    'data_path_2019':'../../input/GLDv2/', #Changed!!
    'valid_csv_fn':'recognition_solution_v2.1.csv',
    'train_csv_fn':'train.csv',
    
#     'gpus':'0,1',
    'gpus':'0,1,2,3', #Changed!!
    'filter_warnings':True,
    'logger': 'neptune',
    'num_sanity_val_steps': 50,

    'distributed_backend': 'ddp',

    'gradient_accumulation_steps':3,
    'precision':16,
    'sync_batchnorm':False,

    'seed':5553,
    'num_workers':4,
    'save_weights_only':True,

    'resume_from_checkpoint': None,
    'pretrained_weights':None,
    'normalization':'imagenet',
    'crop_size':512,

    'backbone':'res2net101_26w_4s',
    'embedding_size': 512,
    'pool': 'gem',
    'arcface_s': 45,
    'arcface_m': 0.4,

    'neck': 'option-D',
    'head':'arc_margin',
    'p_trainable':False,

    'crit': "bce",
    'loss':'arcface',
    #'focal_loss_gamma': 2,
    'class_weights': "log",
    'class_weights_norm' :'batch',
    
    'optimizer': "sgd",
    'weight_decay':1e-4,
    'lr': 0.05,
#     'batch_size': 40,
    'batch_size': 24, #Changed
    'max_epochs': 10,
    'scheduler': {"method":"cosine","warmup_epochs": 1},
    
    'n_classes':81313,
    'data_frac':1.,

    'neptune_project':'tropicbird/kaggle-landmark',#changed!!!
}

args['tr_aug'] = A.Compose([A.Resize(height=544,width=672,p=1.),
    A.RandomCrop(height=args['crop_size'],width=args['crop_size'],p=1.),
    A.HorizontalFlip(p=0.5),
    ])

args['val_aug'] = A.Compose([A.Resize(height=544,width=672,p=1.),
    A.CenterCrop(height=args['crop_size'],width=args['crop_size'],p=1.)
])
