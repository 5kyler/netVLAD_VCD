from multiprocessing import Pool
from tqdm import tqdm
import argparse
import glob
import os

from pymediainfo import MediaInfo

from torchvision import transforms as trn
from torch.utils.data import DataLoader

from VCD import datasets
from VCD.models.local import *
from VCD.utils import *


def parse_metadata(path):
    media_info = MediaInfo.parse(path)
    meta = {'file_path': path}
    for track in media_info.tracks:
        if track.track_type == 'General':
            meta['file_name'] = track.file_name + '.' + track.file_extension
            meta['file_extension'] = track.file_extension
            meta['format'] = track.format
        elif track.track_type == 'Video':
            meta['width'] = int(track.width)
            meta['height'] = int(track.height)
            meta['rotation'] = float(track.rotation or 0.)
            meta['codec'] = track.codec
    return meta


def extract_segment_fingerprint(video, decode_rate, decode_size, transform, cnn_model, group_count, aggr_model):
    # parse video metadata
    meta = parse_metadata(video)
    print(meta)

    # decode all frames
    frames = decode_video_to_pipe(video, meta, decode_rate, decode_size)
    print(len(frames))

    # extract frame fingerprint
    cnn_loader = DataLoader(datasets.ListDataset(frames, transform=transform), batch_size=64, shuffle=False,
                            num_workers=4)

    frame_fingerprints = extract_feature(cnn_model, cnn_loader, progress=False)
    print("extract frame fingerprint: ", frame_fingerprints.shape)

    if group_count != 1:
        k = group_count - frame_fingerprints.shape[0] % group_count
        if k != group_count:
            frame_fingerprints = torch.cat([frame_fingerprints, frame_fingerprints[-1:, ].repeat((k, 1, 1))])

        if aggr_model:
            frame_fingerprints = aggr_model(frame_fingerprints)
            print("aggregating segment feature: ", frame_fingerprints.shape)

        if not aggr_model:
            frame_fingerprints = frame_fingerprints.permute(0, 2, 1)
            frame_fingerprints = frame_fingerprints.reshape(-1, group_count * frame_fingerprints.shape[1],
                                                            frame_fingerprints.shape[-1])
            frame_fingerprints = frame_fingerprints.permute(0, 2, 1)
            print("grouping: ", frame_fingerprints.shape)

    local_features = []
    local_features_set = torch.split(frame_fingerprints, 1)
    for set in local_features_set:
        temp = torch.split(set, 1, dim=2)
        temp = [t.squeeze(-1) for t in temp]
        local_features.append(temp)

    print(f"# of Segments: {len(local_features)}")
    print(f"# of localfeatures per segment: {len(local_features[0])}")
    print(f"shape of localfeature: {local_features[0][0].shape}")
    print("-" * 53)

    return local_features


def load_segment_fingerprint(base_path):
    # base_path
    # ../{dataset}-{decode_rate}-{cnn_extractor}-{group_count}-{aggr_model}/{video}.pth
    # ex) vcdb-5-mobilenet_avg-shot-lstm/00274a.flv.pth

    paths = [os.path.join(base_path, p) for p in os.listdir(base_path)]
    pool = Pool()

    bar = tqdm(range(len(paths)), mininterval=1, ncols=150)
    features = [pool.apply_async(load_file, args=[p], callback=lambda *a: bar.update()) for p in paths]
    pool.close()
    pool.join()
    bar.close()

    features = [f.get() for f in features]
    length = [f.shape[0] for f in features]
    start = np.cumsum([0] + length)
    # index = np.vstack([start[:-1], start[1:]]).reshape(-1, 2)
    index = np.transpose(np.vstack([start[:-1], start[1:]]))
    return np.concatenate(features), np.array(length), index, paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A. extract local features')

    parser.add_argument('--decode_rate', required=False, default=1, help="decode rate")
    parser.add_argument('--decode_size', required=False, default=256, help="decode size")
    parser.add_argument('--group_count', required=False, default=5, help="group count")
    parser.add_argument('--cnn_model', required=True, default='resnet50',
                        help="cnn model (mobilenet, resnet50, intermediate)")
    parser.add_argument('--trained', required=True, type=str2bool, default=False,
                        help="Whether to use the model trained with triplet loss")
    parser.add_argument('--aggr', required=False, type=str2bool, default=False,
                        help="Whether to aggregate frame features")
    parser.add_argument('--feature_path', required=True,
                        default='/nfs_shared_/hkseok/features_local/multiple/resnet50_conv4_5',
                        help="feature path")
    parser.add_argument('--video_dataset', required=False,
                        default='/nfs_shared_/hkseok/VCDB/videos/core/',
                        help="video_dataset path")

    args = parser.parse_args()
    print(args)

    decode_rate = float(args.decode_rate)
    decode_size = int(args.decode_size)
    group_count = int(args.group_count)
    cnn_model = None
    aggr_model = None
    pth_dir = args.feature_path
    if not os.path.isdir(pth_dir):
        os.makedirs(pth_dir)

    if args.cnn_model == 'mobilenet':
        cnn_model = MobileNet_local().cuda()
        if args.trained:
            print("load model...")
            cnn_model.load_state_dict(torch.load('/nfs_shared_/hkseok/mobilenet_avg.pth')['model_state_dict'])
    elif args.cnn_model == 'resnet50':
        cnn_model = Resnet50_local().cuda()
        if args.trained:
            print("load model...")
            cnn_model.load_state_dict(torch.load(
                '/nfs_shared/MLVD/models/resnet_avg_0_10000_norollback_adam_lr_1e-6_wd_0/saved_model/epoch_3_ckpt.pth')[
                                          'model_state_dict'])
    elif args.cnn_model == 'intermediate':
        cnn_model = Resnet50_intermediate().cuda()
        if args.trained:
            print("load model...")
            cnn_model.load_state_dict(torch.load(
                '/nfs_shared/MLVD/models/resnet_avg_0_10000_norollback_adam_lr_1e-6_wd_0/saved_model/epoch_3_ckpt.pth')[
                                          'model_state_dict'])
    cnn_model = nn.DataParallel(cnn_model)

    if args.aggr:
        aggr_model = Local_Maxpooling(group_count)

    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for idx, video in enumerate(glob.glob(os.path.join(args.video_dataset, '*'))):
        videoname = os.path.basename(video)
        print(videoname, idx)
        local_feature = extract_segment_fingerprint(video, decode_rate, decode_size, transform, cnn_model, group_count,
                                                    aggr_model)
        dst_path = os.path.join(pth_dir, videoname + '.pth')
        torch.save(local_feature, dst_path)
