from torchvision.transforms import transforms as trn
from torch.utils.data import DataLoader
import torch

from tqdm import tqdm
import numpy as np
import argparse
import os

from VCD.utils import DEVICE_STATUS, DEVICE_COUNT
from VCD import models
from VCD import datasets


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
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--model', type=str, choices=models.FRAME_MODELS, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=datasets.DATASETS, required=True)
    parser.add_argument('--dataset_root', type=str,required=True)
    parser.add_argument('--feature_path', type=str, required=True)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--worker', type=int, default=4)
    args = parser.parse_args()

    # models
    model = models.get_frame_model(args.model).cuda()

    # Load checkpoints
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    # Check device
    if DEVICE_STATUS and DEVICE_COUNT > 1:
        model = torch.nn.DataParallel(model)

    # Dataset
    dataset = datasets.get_dataset(args.dataset, args.dataset_root)
    if os.path.exists(args.feature_path) and len(os.listdir(args.feature_path)) != 0:
        print(f'Feature Path {args.feature_path} is not empty.')
        exit(1)

    transform = trn.Compose([

        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loader = DataLoader(datasets.ListDataset([], transform=transform), batch_size=args.batch, shuffle=False,
                        num_workers=args.worker)

    extract_frame_features(model, loader, dataset, args.feature_path)