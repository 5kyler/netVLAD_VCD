from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import faiss
import os

from VCD.datasets import FIVR
from VCD.utils import load_feature, l2_distance, find_video_idx
import csv


def write_positive_csv(fivr, positive_csv):
    with open(positive_csv, mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=['a','b'])
        writer.writeheader()
        for query, refers in tqdm(fivr._gt.items()):
            for refer in refers:
                writer.writerow({'a': query, 'b': refer})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate frame triplets for train CNN models with triplet loss.")
    parser.add_argument('--fivr_root', type=str, default='/mldisk/nfs_shared_/MLVD/FIVR')
    parser.add_argument('--feature_path', type=str, default='/mldisk/nfs_shared_/my/NETVLAD/fivr_resnet50_rmac_feature_upgrade')
    parser.add_argument('--triplet_csv', type=str, default='/mldisk/nfs_shared_/my/NETVLAD/triplet_sampling/fivr_triplet_oyyupdate.csv')
    parser.add_argument('--chunk', type=int, default=2000, help='size of negative video pool.')
    parser.add_argument('--margin', type=float, default=0.3, help='distance margin')
    parser.add_argument('--topk', type=int, default=5, help='maximum negative for each positive pair')

    args = parser.parse_args()
    if os.path.exists(args.triplet_csv):
        os.remove(args.triplet_csv)

    positive_csv = os.path.join('/mldisk/nfs_shared_/my/NETVLAD/triplet_sampling/fivr_video_positive.csv')
    fivr = FIVR(args.fivr_root)
    # print(fivr)
    #
    # write_positive_csv(fivr, positive_csv)

    fivr_positive = pd.read_csv(positive_csv).to_numpy()
    core_videos = fivr.core_videos

    np.random.seed(0)
    distract_videos = fivr.distract_videos.copy()
    np.random.shuffle(distract_videos)

    core_feature_path = np.vectorize(lambda x: os.path.join(args.feature_path, x + '.pth'))(core_videos)

    core_feature, core_idx = load_feature(core_feature_path,
                                          progress=True,
                                          desc='Load core video features')

    distract_feature_paths = np.vectorize(lambda x: os.path.join(args.feature_path, x + '.pth'))(distract_videos)

    all_triplets = []
    for i in range(0, len(distract_videos), args.chunk):
        distract_video = distract_videos[i:i + args.chunk]
        distract_feature_path = distract_feature_paths[i:i + args.chunk]
        distract_feature, distract_idx = load_feature(distract_feature_path,
                                                      progress=True,
                                                      desc='Load distract video features')
        # load_feature 에서 load_file2 함수 바꿈 !!!!!!!!!!!!
        # distract_feature = np.stack([np.mean(distract_feature[m[0]:m[1]], axis=0) for m in distract_idx])

        bg_index = faiss.IndexFlatL2(distract_feature.shape[1])

        bg_index = faiss.index_cpu_to_all_gpus(bg_index)
        bg_index.add(distract_feature)

        used = set()

        # frame triplets
        # fine negative features (dist(a,b) - margin < dist(a, n) < dist(a,b))
        triplets = []
        for j, pair in enumerate(tqdm(fivr_positive, ncols=150, desc='Triplets Sampling', unit='pair'), start=1):
            a, b = pair
            # find feature a
            a_video_idx = find_video_idx(a, fivr.core_videos)
            a_feature_idx = core_idx[a_video_idx][0]
            a_feature_end_idx = core_idx[a_video_idx][1]
            # a_feature = core_feature[a_feature_idx:a_feature_idx + a_feature_end_idx, :]
            a_feature = core_feature[a_feature_idx:a_feature_end_idx]

            # find feature b
            b_video_idx = find_video_idx(b, fivr.core_videos)
            b_feature_idx = core_idx[b_video_idx][0]
            b_feature_end_idx = core_idx[b_video_idx][1]
            # b_feature = core_feature[b_feature_idx:b_feature_idx + b_feature_end_idx, :]
            b_feature = core_feature[b_feature_idx:b_feature_end_idx]

            # # average pooling frame features
            # a_feature = np.mean(a_feature, axis=0).reshape(1,-1)
            # b_feature = np.mean(b_feature, axis=0).reshape(1,-1)

            # dist(a,b)
            pos_distance = l2_distance(a_feature, b_feature)

            if pos_distance != 0:
                neg_distance, neg_rank = bg_index.search(np.concatenate([a_feature, b_feature]), 1024)  # (2, distract)

                # compare A - negative pool
                a_n_distance, a_n_rank = neg_distance[0], neg_rank[0]
                valid_neg_rank = np.where((pos_distance < a_n_distance + args.margin) &
                                          (a_n_distance < pos_distance))[0]
                neg_idx = [a_n_rank[r] for r in valid_neg_rank if a_n_rank[r] not in used][:args.topk]

                triplets += [{'anchor': a,
                              'positive': b,
                              'negative': os.path.join(distract_video[i])} for i in neg_idx]

                used.update(set(neg_idx))

                # compare B - negative pool
                b_n_distance, b_n_rank = neg_distance[1], neg_rank[1]
                valid_neg_rank = np.where((pos_distance - args.margin < b_n_distance) &
                                          (b_n_distance < pos_distance))[0]
                neg_idx = [b_n_rank[r] for r in valid_neg_rank if b_n_rank[r] not in used][:args.topk]

                triplets += [{'anchor': b,
                              'positive': a,
                              'negative': os.path.join(distract_video[i])} for i in neg_idx]
                used.update(set(neg_idx))

        del bg_index
        del distract_feature
        # if not os.path.exists(args.triplet_csv):
        #     pd.DataFrame(triplets).to_csv(args.triplet_csv, mode='w', index=False)
        # else:
        #     pd.DataFrame(triplets).to_csv(args.triplet_csv, mode='a', index=False, header=False)

        all_triplets += triplets
    print(len(all_triplets))
    pd.DataFrame(all_triplets).to_csv(args.triplet_csv, index=False)