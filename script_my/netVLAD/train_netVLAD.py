from tqdm import tqdm
import numpy as np
import argparse
import warnings
import os

from collections import defaultdict
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
import torch
import torch_optimizer
import faiss

from VCD import models
from VCD import datasets
from VCD.utils import *

from script_my.netVLAD.netvlad import NetVLAD, NetVLADPCA

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = None
writer = None


@torch.no_grad()
def extract_segment_feature(model_, model, feature_path, **kwargs):
    global count
    model.eval()
    if model_ == 'rmac':
        count = 5
    elif model_ == 'semi_rmac':
        count = 70

    segment = []
    for p in tqdm(feature_path, ncols=150, unit='video'):
        frame_feature = torch.load(p)
        k = count - frame_feature.shape[0] % count
        if k != count:
            frame_feature = torch.cat([frame_feature, frame_feature[-1:, ].repeat((k, 1))])
        frame_feature = frame_feature.reshape(-1, count, frame_feature.shape[-1])
        frame_feature = frame_feature.transpose(1,2).unsqueeze(3)
        segment.append(frame_feature)
    segment_set = torch.cat(segment)

    length = [seg.shape[0] for seg in segment]
    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    # 모델에 집어넣기
    segment_set_1 = model(segment_set[:9799].cuda(), **kwargs).cpu()
    segment_set_2 = model(segment_set[9799:].cuda(), **kwargs).cpu()
    segment_set = torch.cat([segment_set_1, segment_set_2])

    del segment
    del segment_set_1
    del segment_set_2

    return np.array(segment_set), np.array(length), index


@torch.no_grad()
def scan_vcdb_annotation(root):
    def parse(ann):
        a, b, *times = ann.strip().split(',')
        times = [sum([60 ** (2 - n) * int(u) for n, u in enumerate(t.split(':'))]) for t in times]
        return [a, b, *times]

    groups = os.listdir(root)
    annotations = []

    for g in groups:
        f = open(os.path.join(root, g), 'r')
        annotations += [[os.path.splitext(g)[0], *parse(l)] for l in f.readlines()]

    return annotations


@torch.no_grad()
def eval_topk_300_recall(net, test_feature_base, epoch):
    np.set_printoptions(threshold=3, edgeitems=3)
    vcdb_videos = np.load('/mldisk/nfs_shared/MLVD/VCDB/meta/vcdb_videos_core.npy')
    feature_path = np.char.add(np.char.add(test_feature_base + '/', vcdb_videos), '.pth')
    segment_annotation = scan_vcdb_annotation('/mldisk/nfs_shared/MLVD/VCDB/annotation')
    video2id = {v: n for n, v in enumerate(vcdb_videos)}

    model_type = 'rmac' # or semi_rmac

    segment_features, length, location = extract_segment_feature(model_type, net, feature_path, single=True)

    table = dict()
    count = 0
    for video_idx, ran in enumerate(location):
        for features_idx in range(ran[1] - ran[0]):
            table[count] = (video_idx, features_idx)
            count += 1

    db_interval = dict()
    for n, v in enumerate(vcdb_videos):
        vid = video2id[v]
        db_interval[v] = [[i * 5, (i + 1) * 5] for i in range(0, length[vid])]  # 0.2fps

    feature_annotation = defaultdict(list)
    for ann in segment_annotation:
        g, a, b, sa, ea, sb, eb = ann
        ai = [n for n, i in enumerate(db_interval[a]) if not (i[1] <= sa or ea <= i[0])]
        bi = [n for n, i in enumerate(db_interval[b]) if not (i[1] <= sb or eb <= i[0])]

        cnt = len(ai)
        af = np.linspace(ai[0], ai[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
        bf = np.linspace(bi[0], bi[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
        feature_annotation[a].append([b, np.concatenate([af, bf], axis=1)])
        if a != b:
            cnt = len(bi)
            af = np.linspace(ai[0], ai[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
            bf = np.linspace(bi[0], bi[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
            feature_annotation[b].append([a, np.concatenate([bf, af], axis=1)])

    index = faiss.IndexFlatIP(segment_features.shape[1])
    faiss.normalize_L2(segment_features)
    index.add(segment_features)

    result = dict()
    for qv_idx, qv in enumerate(vcdb_videos):
        ann = feature_annotation[qv]
        qv_feature = segment_features[location[qv_idx][0]:location[qv_idx][1]]
        # print(qv_idx, qv, qv_feature.shape, length[qv_idx])
        D, I = index.search(qv_feature, segment_features.shape[0])
        result[qv] = defaultdict(list)
        for a in ann:
            loc = location[video2id[a[0]]]
            query_time = a[1][:, 0]
            ref_idx = a[1][:, 1] + loc[0]
            rank = [np.where(np.abs(I[t, :] - ref_idx[n]) <= 2)[0][0] for n, t in enumerate(query_time)]
            ret = np.vstack([a[1][:, 0], a[1][:, 1], rank]).T
            result[qv][a[0]].append(ret)
    out = []
    recall = []

    result_per_feature = dict()
    for qv, ret in result.items():
        result_per_feature[qv] = defaultdict(list)
        for rf, ranks in ret.items():
            for r in ranks:
                for i in r:
                    result_per_feature[qv][i[0]].append(i[2])
                    out.append(i[2])
    out = np.array(out)
    total = out.shape[0]

    a = 1
    r = np.where(out < a)[0].shape[0]

    recall.append({'topk': a, 'recall': r / total, 'count': r})
    get_rank = []
    for a in range(10, 500, 10):
        r = np.where(out < a)[0].shape[0]
        recall.append({'topk': a, 'recall': r / total, 'count': r})
        if a == 300:
            rec = r / total
            get_rank.append(a)
            get_rank.append(rec)

    logger.info(f'[Epoch {epoch}] '
                f'topk: {get_rank[0]:2f}, recall :{get_rank[1]:4f}')

    writer.add_scalar('test/topk', get_rank[0], epoch)
    writer.add_scalar('test/recall', get_rank[1], epoch)

    del feature_annotation
    del result
    del result_per_feature
    del segment_features
    del recall
    torch.cuda.empty_cache()


def make_train_batch(samples):
    anc = [sample['anchor'] for sample in samples]
    pos = [sample['positive'] for sample in samples]
    neg = [sample['negative'] for sample in samples]

    anc = torch.nn.utils.rnn.pad_sequence(anc, batch_first=True)
    anc = torch.unsqueeze(anc.transpose(2, 1), dim=-1)

    pos = torch.nn.utils.rnn.pad_sequence(pos, batch_first=True)
    pos = torch.unsqueeze(pos.transpose(2, 1), dim=-1)

    neg = torch.nn.utils.rnn.pad_sequence(neg, batch_first=True)
    neg = torch.unsqueeze(neg.transpose(2, 1), dim=-1)

    return {'anchor': anc.contiguous(), 'positive': pos.contiguous(), 'negative': neg.contiguous()}


def train(net, loader, optimizer, criterion, scheduler, epoch):
    losses = AverageMeter()
    net.train()
    bar = tqdm(loader, ncols=150)
    for i, data in enumerate(loader, 1):
        optimizer.zero_grad()
        _output1, _output2, _output3 = net(data['anchor'].cuda(), data['positive'].cuda(), data['negative'].cuda())
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


def make_test_batch(samples):
    path = [sample['path'] for sample in samples]
    feature = [sample['feature'] for sample in samples]

    feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True)
    feature = torch.unsqueeze(feature.transpose(2, 1), dim=-1)

    return {'feature': feature.contiguous(), 'path': path}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rmac + NetVLAD + TripletLoss")
    parser.add_argument('--model', type=str, choices=models.FRAME_MODELS, required=False, default='Resnet50_RMAC')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--triplet_csv', type=str, required=False, default='/workspace/script_my/netVLAD/csv_folder/apn_test.csv')
    parser.add_argument('--fivr_root', type=str, required=False, default='/mldisk/nfs_shared_/MLVD/FIVR')
    parser.add_argument('--vcdb_root', type=str, required=False, default='/mldisk/nfs_shared_/MLVD/VCDB-core')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-m', '--margin', type=float, default=0.3)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('-tb', '--test_batch', type=int, default=64)
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
    netvlad = NetVLAD(num_clusters=16, dim=2048, alpha=1.0).cuda()
    embed_net = NetVLADPCA(netvlad, 8192).cuda()

    # Load checkpoints
    if args.ckpt is not None:
        embed_net.load_state_dict(torch.load(args.ckpt))

    model = models.TripletNet(embed_net).cuda()
    writer.add_graph(model, [torch.rand((2, 2048, 1, 1)).cuda(),
                             torch.rand((2, 2048, 1, 1)).cuda(),
                             torch.rand((2, 2048, 1, 1)).cuda()])
    logger.info(model)

    if DEVICE_STATUS and DEVICE_COUNT > 1:
        model = torch.nn.DataParallel(model.to('cuda:0'))

    train_feature_path = '/mldisk/nfs_shared_/my/NETVLAD/fivr_resnet50_rmac_feature'
    test_feature_path = '/mldisk/nfs_shared_/my/NETVLAD/vcdb_resnet50_rmac_feature'

    train_loader = DataLoader(datasets.TripletfeatureDataset(args.triplet_csv, train_feature_path),
                              collate_fn=make_train_batch, shuffle=True, num_workers=args.worker, batch_size=args.batch)

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

    # eval_topk_300_recall(model, test_feature_path, 1)
    print("train 하러가자!")
    for e in range(1, args.epoch, 1):
        train(model, train_loader, optimizer, criterion, scheduler, e)
        eval_topk_300_recall(model, test_feature_path, e)

        if args.scheduler != 'cyclic':
            scheduler.step()

        torch.save({'epoch': e,
                    'state_dict': model.module.embedding_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(model_save_dir, f'epoch_{e}_ckpt.pth'))
