from tqdm import tqdm
import numpy as np
import argparse
import warnings
import os

from torchvision.transforms import transforms as trn
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torch_optimizer

from VCD import models
from VCD import datasets
from VCD.utils import *

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = None
writer = None


@torch.no_grad()
def eval_vcdb_frame_rank(net, loader, vcdb, epoch, chunk=1000, margin=1):
    loader.dataset.l = [f for v in vcdb.core_videos for f in vcdb.get_frames(v)]
    features = extract_feature(net, loader, progress=True, single=True)
    length = [vcdb.get_framecount(v) for v in vcdb.core_videos]
    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    ranks = vcdb_frame_retrieval(vcdb, features, index, chunk, margin, progress=True)
    avg_rank, avg_dist, avg_rank_per_query, avg_dist_per_query = ranks
    logger.info(f'[Epoch {epoch}] '
                f'Rank/Frame: {avg_rank:4f}, Rank/Seg :{avg_rank_per_query:4f}, '
                f'Distance/Frame: {avg_dist:4f}, Distance/Seg: {avg_dist_per_query:4f}')

    writer.add_scalar('test/rank_per_frame', avg_rank, epoch)
    writer.add_scalar('test/distance_per_frame', avg_dist, epoch)
    writer.add_scalar('test/rank_per_query', avg_rank_per_query, epoch)
    writer.add_scalar('test/distance_per_query', avg_dist_per_query, epoch)

    del features


def train(net, loader, optimizer, criterion, scheduler, epoch):
    losses = AverageMeter()
    net.train()
    bar = tqdm(loader, ncols=150)
    for i, (path, frames) in enumerate(loader, 1):
        optimizer.zero_grad()
        _output1, _output2, _output3 = net(frames[0].cuda(), frames[1].cuda(), frames[2].cuda())
        loss = criterion(_output1, _output2, _output3)
        losses.update(loss.item())
        loss.backward()
        optimizer.step()
        if scheduler.__class__.__name__ == 'CyclicLR':
            scheduler.step()

        bar.set_description(f'[Epoch {epoch}] '
                            f'lr: {scheduler.get_lr()[0]:4f}, '
                            f'loss: {losses.val:.4f}({losses.avg:.4f}), ')
        bar.update()

    bar.close()

    logger.info(f'[EPOCH {epoch}] '
                f'lr: {scheduler.get_lr()[0]:4f}, '
                f'loss: {losses.avg:4f}, ')
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/learning_rate', scheduler.get_lr()[0], epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN model.")
    parser.add_argument('--model', type=str, choices=models.FRAME_MODELS, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--triplet_csv', type=str, required=True)
    parser.add_argument('--fivr_root', type=str, required=True)
    parser.add_argument('--vcdb_root', type=str, required=True)

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-m', '--margin', type=float, default=0.3)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-tb', '--test_batch', type=int, default=256)
    parser.add_argument('-w', '--worker', type=int, default=4)
    parser.add_argument('-o', '--optim', type=str, default='radam')
    parser.add_argument('-s', '--scheduler', type=str, default='cyclic')

    parser.add_argument('-l', '--log_dir', type=str, default='./log')
    parser.add_argument('-c', '--comment', type=str, default=None)
    args = parser.parse_args()

    writer, logger, log_dir = initialize_writer_and_log(args.log_dir, args.comment)
    model_save_dir = os.path.join(log_dir, 'saved_model')
    os.makedirs(model_save_dir)
    logger.info(args)

    # Model
    embed_net = models.get_frame_model(args.model).cuda()
    # Load checkpoints
    if args.ckpt is not None:
        embed_net.load_state_dict(torch.load(args.ckpt))

    model = models.TripletNet(embed_net).cuda()
    writer.add_graph(model, [torch.rand((2, 3, 224, 224)).cuda(),
                             torch.rand((2, 3, 224, 224)).cuda(),
                             torch.rand((2, 3, 224, 224)).cuda()])
    logger.info(model)

    if DEVICE_STATUS and DEVICE_COUNT > 1:
        model = torch.nn.DataParallel(model)

    # Data
    transform = {
        'train': A.Compose([
            A.Resize(256, 256),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=224, width=224),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.RandomRotate90(p=1),
                A.VerticalFlip(p=1)
            ], p=.5),
            A.OneOf([
                A.MotionBlur(p=1),
                A.OpticalDistortion(p=1),
                A.GaussNoise(p=1)
            ], p=.5),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.1),
            A.ToGray(p=0.05),
            A.IAACropAndPad(p=0.3),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0)
        ]),
        'test': trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    vcdb = datasets.get_dataset('VCDB', args.vcdb_root)
    fivr = datasets.get_dataset('FIVR', args.fivr_root)

    train_loader = DataLoader(datasets.TripletDataset(args.triplet_csv, fivr.frame_root, transform=transform['train']),
                              shuffle=True, num_workers=args.worker, batch_size=args.batch)
    test_loader = DataLoader(datasets.ListDataset([], transform=transform['test']),
                             shuffle=False, num_workers=args.worker, batch_size=args.test_batch)

    # Optimizer
    criterion = nn.TripletMarginLoss(args.margin)
    l2_dist = nn.PairwiseDistance()

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'radam':
        optimizer = torch_optimizer.RAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    if args.scheduler == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.learning_rate, args.learning_rate * 5,
                                                      step_size_up=500, step_size_down=500,
                                                      mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                      cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
                                                      last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

    eval_vcdb_frame_rank(model, test_loader, vcdb, 0, 1000, 1)
    for e in range(1, args.epoch, 1):
        train(model, train_loader, optimizer, criterion, scheduler, e)
        eval_vcdb_frame_rank(model, test_loader, vcdb, e, 1000, 1)

        if args.scheduler != 'cyclic':
            scheduler.step()

        torch.save({'epoch': e,
                    'state_dict': model.module.embedding_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(model_save_dir, f'epoch_{e}_ckpt.pth'))
