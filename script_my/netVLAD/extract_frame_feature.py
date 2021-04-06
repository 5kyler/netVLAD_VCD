from torchvision.transforms import transforms as trn
from torch.utils.data import DataLoader
import torch

from tqdm import tqdm

import os

from VCD.utils import DEVICE_STATUS, DEVICE_COUNT
from VCD import models
from VCD import datasets


@torch.no_grad()
def extract_semi_frame_features(model, loader, dataset, save_to):
    model.eval()
    videos = dataset.all_videos
    frames = [f for v in videos for f in dataset.get_frames(v)]
    loader.dataset.l = frames
    bar = tqdm(loader, ncols=150, unit='batch')
    features = []

    vidx = 0
    for idx, (paths, frames) in enumerate(bar):
        feat = model(frames.cuda()).cpu()
        features.append(feat)
        features = torch.cat(features)
        while vidx < len(videos):
            c = dataset.get_framecount(videos[vidx])
            re_c = c*14
            if features.shape[0] >= re_c:
                target = os.path.join(save_to, f'{videos[vidx]}.pth')
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                torch.save(features[:re_c, ], target)
                bar.set_description_str(os.path.basename(target))
                features = features[re_c:, ]
                vidx += 1
            else:
                break
        features = [features]


@torch.no_grad()
def extract_frame_features(model, loader, dataset, save_to):
    model.eval()
    videos = dataset.all_videos
    frames = [f for v in videos for f in dataset.get_frames(v)]
    loader.dataset.l = frames
    bar = tqdm(loader, ncols=150, unit='batch')
    features = []

    vidx = 0
    for idx, (paths, frames) in enumerate(bar):
        feat = model(frames.cuda()).cpu()
        features.append(feat)
        features = torch.cat(features)
        while vidx < len(videos):
            c = dataset.get_framecount(videos[vidx])
            if features.shape[0] >= c:
                target = os.path.join(save_to, f'{videos[vidx]}.pth')
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                torch.save(features[:c, ], target)
                bar.set_description_str(os.path.basename(target))
                features = features[c:, ]
                vidx += 1
            else:
                break
        features = [features]



if __name__ == '__main__':
    # models
    model = models.get_frame_model('Resnet50_RMAC').cuda()

    ckpt = ''
    # Load checkpoints
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))

    # Check device
    if DEVICE_STATUS and DEVICE_COUNT > 1:
        model = torch.nn.DataParallel(model)
    feature_path = '/mldisk/nfs_shared_/my/NETVLAD/vcdb_resnet50_semi_rmac_feature'
    os.makedirs(feature_path)
    # Dataset
    dataset = datasets.get_dataset('VCDB', '/mldisk/nfs_shared_/MLVD/VCDB-core')


    transform = trn.Compose([

        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loader = DataLoader(datasets.ListDataset([], transform=transform), batch_size=256, shuffle=False,
                        num_workers=4)

    extract_frame_features(model, loader, dataset, feature_path)