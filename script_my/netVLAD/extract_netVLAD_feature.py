import torch.nn as nn
import os.path

from script_my.netVLAD.netvlad import NetVLAD, NetVLADPCA

from torchvision.transforms import transforms as trn
from torch.utils.data import DataLoader
import torch

from tqdm import tqdm
import os

from VCD import datasets
from VCD import models
from VCD.utils import DEVICE_STATUS, DEVICE_COUNT


# netVLAD 피쳐 뽑는 코드.... 모델 저장한거 불러와~

@torch.no_grad()
def extract_frame_features(model, loader, dataset, save_to, **kwargs):
    model.eval()
    videos = dataset.all_videos
    # frames = [f for v in videos for f in dataset.get_frames(v)]
    loader.dataset.l = videos
    bar = tqdm(loader, ncols=150, unit='batch')

    for idx, data in enumerate(bar):
        features = model(data['feature'].cuda(), **kwargs).cpu()
        save_path = data['save_path']

        for i in range(len(save_path)):
            target = os.path.join(save_to, f'{save_path[i]}.pth')
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            torch.save(features[i], target)
            bar.set_description_str(os.path.basename(target))


def collate_fn(samples):
    feature = [sample['feature'] for sample in samples]
    path = [sample['path'] for sample in samples]
    save_path = [sample['save_path'] for sample in samples]

    feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True)
    feature = torch.unsqueeze(feature.transpose(2, 1), dim=-1)

    return {"path": path, 'feature': feature.contiguous(), "save_path": save_path}


if __name__ == '__main__':
    # model
    netvlad = NetVLAD(num_clusters=16, dim=2048, alpha=1.0).cuda()
    embed_net = NetVLADPCA(netvlad, 8192).cuda()

    # check point
    ckpt_path = '/workspace/script_my/netVLAD/log/20210330/131215/saved_model/epoch_1_ckpt.pth'
    ckpt = torch.load(ckpt_path)
    ckpt_state_dict = ckpt['state_dict']
    embed_net.load_state_dict(ckpt_state_dict)

    model = models.TripletNet(embed_net).cuda()

    # Dataset
    train_feature_path = '/mldisk/nfs_shared_/my/NETVLAD/fivr_resnet50_rmac_feature'
    dataset = datasets.get_dataset('FIVR', '/mldisk/nfs_shared_/MLVD/FIVR')
    loader = DataLoader(datasets.ListfeatureDataset([], train_feature_path), batch_size=64, collate_fn=collate_fn, shuffle=False,
                        num_workers=4)

    feature_path = '/mldisk/nfs_shared_/my/NETVLAD/fivr_resnet50_rmac_feature_upgrade'
    #os.makedirs(feature_path)
    extract_frame_features(model, loader, dataset, feature_path, single=True)





