import sys
import importlib
from types import SimpleNamespace
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import softmax
from joblib import Parallel, delayed
import seaborn as sns

sys.path.append("../src")
sys.path.append("../configs")

sys.argv = ['--config', 'config7']

from models import *
from loss import *
from train import *
from data import *
import numpy as np

def gpu_unravel(batch):
    input_dict, target_dict = batch
    input_dict = {k: input_dict[k].cuda() for k in input_dict}
    target_dict = {k: target_dict[k].cuda() for k in target_dict}
    return input_dict, target_dict

def get_embeddings(dl, model):
    with torch.no_grad():
        embeddings = np.zeros((len(dl.dataset) , 512))
        total = len(dl)
#         target_list = [] #Added!!!
        for idx, batch in tqdm(enumerate(dl), total=len(dl)):
            input_dict, target_dict = dict_unravel(batch)

            outs = model.forward(input_dict, get_embeddings=True)["embeddings"]

#             target_list.append(target_dict['target']) #Added!!!
            embeddings[idx*batch_size:idx*batch_size+outs.size(0),:] = outs.detach().cpu().numpy()

#     return embeddings, target_list #Changed!!!
    return embeddings

if __name__ == '__main__':

    dict_unravel = gpu_unravel

    name = "config7"
    # pretrained_weights = "../models/config1_ckpt_10.pth"
    pretrained_weights = "../models/config7/config7_ckpt_10.pth"

    csv = "train"

    #このtrainはGLRv2のfull trainだと思う。
    # train = pd.read_csv(f"../embeddings/{csv}.csv")
    train = pd.read_csv(f"../../input/GLDv2/{csv}.csv")

    # train["img_folder"] = "/ssd/kaggle-landmark/input/train/"
    train["img_folder"] = "../../input/GLDv2/train/"
    # train["img_folder"] = "/home/ubuntu/kaggle/input/GLDv2/train/"
    train["target"] = 0
#     train=train.rename(columns={'landmark_id': 'target'})

    train=train[['id','img_folder','target']]
    
#     train=train[:5120]
    print(f"train size: {len(train)}")

#     aug = A.Compose([
#                     A.SmallestMaxSize(512),
#                     A.CenterCrop(always_apply=False, p=1.0, height=512, width=512),
#                     ],
#                     p=1.0
#                     )

    
    aug = A.Compose([
                    A.Resize(height=544,width=672,p=1.),
                    A.CenterCrop(always_apply=False, height=544,width=672,p=1.),
                    ],
                    p=1.0)
    
    
    

    val_ds = GLRDataset(train, normalization=args.normalization, aug=aug)

    batch_size = 384
    print(f'batch_size {batch_size}')
    #batch_size = 8
    nw=24
    print(f'num_workers {nw}')
    val_dl = DataLoader(dataset=val_ds,
                        batch_size=batch_size,
                        sampler=SequentialSampler(val_ds), collate_fn=collate_fn, num_workers=nw, pin_memory=True)

    print("load model")
    model = Net(args)
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load(pretrained_weights))
    model = nn.DataParallel(model)
    for i in tqdm(range(1000)):
        pass
    print("get embeddings")
    #landmark_id to calss_idは不要か。
#     embeddings, target_list = get_embeddings(val_dl, model) #Changed!!
    embeddings = get_embeddings(val_dl, model)
    #print(f"shape of embeddings:{embeddings.shape}")
    print("saving the embeddings")
    np.save(f"../embeddings/{name}_{csv}_embeddings", embeddings)
    print("saving is done")
    
#     #Added!!!!
#     import pickle
#     print("saving the target_lists")
#     with open(f"../embeddings/{name}_{csv}_target_list_TEST.pkl", 'wb') as f:
#         pickle.dump(target_list, f)
#     print("saving is done")
