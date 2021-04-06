from torchvision.transforms import transforms as trn
from torch.utils.data import DataLoader
import torch

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import argparse
import faiss
import os

from VCD import models
from VCD.datasets import VCDB, ListDataset
from VCD.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    subparsers = parser.add_subparsers()

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--vcdb_root', type=str, required=True)
    common.add_argument('--chunk', type=int, default=100)
    common.add_argument('--margin', type=int, default=1)

    extract = subparsers.add_parser('extract', parents=[common],
                                    help='Extract features and Evaluate VCDB frame ranking')
    extract.add_argument('--model', type=str, choices=models.FRAME_MODELS)
    extract.add_argument('--ckpt', type=str, default=None)
    extract.add_argument('--batch', type=int, default=256)
    extract.add_argument('--worker', type=int, default=4)

    parser_load = subparsers.add_parser('load', parents=[common],
                                        help='Load features and Evaluate VCDB frame ranking')
    parser_load.add_argument('--feature_path', type=str, required=True)
    args = parser.parse_args()
    print(args)
    if not hasattr(args, 'model') and not hasattr(args, 'model'):
        parser.print_help()
        exit(-1)

    vcdb = VCDB(args.vcdb_root)
    if hasattr(args, 'model'):

        loader = DataLoader(ListDataset([f for v in vcdb.core_videos for f in vcdb.get_frames(v)]),
                            batch_size=args.batch, shuffle=False, num_workers=args.worker)

        model = models.get_frame_model(args.model).cuda()
        # Load checkpoints
        if args.ckpt is not None:
            model.load_state_dict(torch.load(args.ckpt))

        # Check device
        if DEVICE_STATUS and DEVICE_COUNT > 1:
            model = torch.nn.DataParallel(model)

        features = extract_feature(model, loader)
        length = [vcdb.get_framecount(v) for v in vcdb.core_videos]
        start = np.cumsum([0] + length)
        index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    else:
        features, index = load_feature([os.path.join(args.feature_path, v + '.pth')
                                        for v in vcdb.core_videos], progress=False)

    vcdb_frame_retrieval(vcdb, features, index, args.chunk, args.margin)
