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

sys.argv = ['--config', 'config1']

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
        for idx, batch in tqdm(enumerate(dl), total=len(dl)):
            input_dict, target_dict = dict_unravel(batch)

            outs = model.forward(input_dict, get_embeddings=True)["embeddings"]

            embeddings[idx*batch_size:idx*batch_size+outs.size(0),:] = outs.detach().cpu().numpy()

    return embeddings

if __name__ == '__main__':

    dict_unravel = gpu_unravel

    name = "config1"
    # pretrained_weights = "../models/config1_ckpt_10.pth"
    pretrained_weights = "../models/config1/config1_ckpt_10.pth"

    csv = "train"

    #このtrainはGLRv2のfull trainだと思う。
    # train = pd.read_csv(f"../embeddings/{csv}.csv")
    train = pd.read_csv(f"../../input/GLDv2/{csv}.csv")

    # train["img_folder"] = "/ssd/kaggle-landmark/input/train/"
    train["img_folder"] = "../../input/GLDv2/train/"
    # train["img_folder"] = "/home/ubuntu/kaggle/input/GLDv2/train/"
    train["target"] = 0

    train=train[['id','landmark_id','img_folder','target']]

    aug = A.Compose([
                    A.SmallestMaxSize(512),
                    A.CenterCrop(always_apply=False, p=1.0, height=512, width=512),
                    ],
                    p=1.0
                    )


    val_ds = GLRDataset(train, normalization=args.normalization, aug=aug)

    batch_size = 512
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
    embeddings = get_embeddings(val_dl, model)
    #print(f"shape of embeddings:{embeddings.shape}")
    print("saving the embeddings")
    np.save(f"../embeddings/{name}_{csv}_embeddings", embeddings)
    print("saving is done")
