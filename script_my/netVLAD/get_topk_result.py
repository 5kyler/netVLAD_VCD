import os
import numpy as np
from multiprocessing import Pool
import torch
import tqdm
import faiss
from collections import defaultdict
import pickle as pk
import pandas as pd
import shutil


def make_directory(path, move_path):
    os.makedirs(move_path)
    for r, d, f in os.walk(path):
        for file in f:
            full = os.path.join(r, file)
            shutil.copy(full, move_path)
    print('폴더 생성 및 랭킹계산 피쳐 저장')


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


def load(path):
    feat = torch.load(path)
    return feat


def load_features(paths):
    pool = Pool()
    bar = tqdm.tqdm(range(len(paths)), mininterval=1, ncols=150)
    features = [pool.apply_async(load, args=[path], callback=lambda *a: bar.update()) for path in paths]
    pool.close()
    pool.join()
    bar.close()
    features = [f.get() for f in features]
    length = [f.shape[0] for f in features]

    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    return np.concatenate(features), np.array(length), index


def get_result1(name, feature_base):
    np.set_printoptions(threshold=3, edgeitems=3)
    vcdb_videos = np.load('/workspace/for_process/vcdb_videos_core.npy')
    # segment_annotation, feature_annotation = scan_vcdb_annotation ('/MLVD/VCDB/annotation')
    segment_annotation = scan_vcdb_annotation('/mldisk/nfs_shared_/MLVD/VCDB/annotation')
    video2id = {v: n for n, v in enumerate(vcdb_videos)}
    feature_base = feature_base

    name = name

    feature_path = np.char.add(np.char.add(feature_base+'/', vcdb_videos), '.pth')
    feature, length, location = load_features(feature_path)
    # print(feature_annotation)

    table = dict()
    count = 0
    for video_idx, ran in enumerate(location):
        for features_idx in range(ran[1] - ran[0]):
            table[count] = (video_idx, features_idx)
            count += 1

    mapping = np.vectorize(lambda x, table: table[x])
    print(table)

    db_interval = dict()
    for n, v in enumerate(vcdb_videos):
        vid = video2id[v]
        # db_interval[v] = [[i * 5, (i + 1) * 5] for i in range(0, length[vid])] # 0.2fps
        db_interval[v] = [[i, (i + 1)] for i in range(0, length[vid])] # 1fps

    feature_annotation=defaultdict(list)
    for ann in segment_annotation:
        g, a, b, sa, ea, sb, eb = ann
        ai = [n for n, i in enumerate(db_interval[a]) if not (i[1] <= sa or ea <= i[0])]
        bi = [n for n, i in enumerate(db_interval[b]) if not (i[1] <= sb or eb <= i[0])]

        cnt=len(ai)
        af = np.linspace(ai[0], ai[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
        bf = np.linspace(bi[0], bi[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
        feature_annotation[a].append([b, np.concatenate([af, bf], axis=1)])
        if a!=b:
            cnt = len(bi)
            af = np.linspace(ai[0], ai[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
            bf = np.linspace(bi[0], bi[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
            feature_annotation[b].append([a, np.concatenate([bf, af], axis=1)])



    index = faiss.IndexFlatIP(feature.shape[1])
    faiss.normalize_L2(feature)
    index.add(feature)

    result = dict()
    for qv_idx, qv in enumerate(vcdb_videos):
        ann = feature_annotation[qv]
        qv_feature = feature[location[qv_idx][0]:location[qv_idx][1]]
        print(qv_idx, qv, qv_feature.shape, length[qv_idx])
        D, I = index.search(qv_feature, feature.shape[0])
        result[qv] = defaultdict(list)
        for a in ann:
            loc = location[video2id[a[0]]]
            query_time = a[1][:, 0]
            ref_idx = a[1][:, 1] + loc[0]
            rank = [np.where(np.abs(I[t, :] - ref_idx[n]) <= 2)[0][0] for n, t in enumerate(query_time)]
            ret = np.vstack([a[1][:, 0], a[1][:, 1], rank]).T
            result[qv][a[0]].append(ret)

    # print([ for k,v in result['00274a923e13506819bd273c694d10cfa07ce1ec.flv'] for vv in v])

    pk.dump(result, open(f'/workspace/test2/{name}.pkl', 'wb'))
    result = pk.load(open(f'/workspace/test2/{name}.pkl', 'rb'))
    # print(result)
    result_per_feature = dict()
    for qv, ret in result.items():
        result_per_feature[qv] = defaultdict(list)
        for rf, ranks in ret.items():
            for r in ranks:
                for i in r:
                    result_per_feature[qv][i[0]].append(i[2])
    print(result_per_feature)


def get_result2(name):
    result = pk.load(open(f'/workspace/test2/{name}.pkl', 'rb'))

    # print(result)

    out = []

    recall_csv = f'/workspace/test2/{name}-recall.csv'
    recall = []
    histogram = f'/workspace/test2/{name}-histogram.csv'

    result_per_feature = dict()
    for qv, ret in result.items():
        result_per_feature[qv] = defaultdict(list)
        for rf, ranks in ret.items():
            for r in ranks:
                for i in r:
                    result_per_feature[qv][i[0]].append(i[2])
                    out.append(i[2])
    # print(result_per_feature)
    out = np.array(out)
    print(out)
    total = out.shape[0]

    max = np.max(out)
    print(max)

    a = 1
    r = np.where(out < a)[0].shape[0]
    # print(f'recall at {a} : {r/total:.4f} {r}/{total}')
    recall.append({'topk': a, 'recall': r / total, 'count': r})
    print(r)
    for a in range(10, 1000, 10):
        r = np.where(out < a)[0].shape[0]
        # print(f'recall at {a} : {r/total:.4f} {r}/{total}')
        recall.append({'topk': a, 'recall': r / total, 'count': r})

    for a in range(1000, 10000, 1000):
        r = np.where(out < a)[0].shape[0]
        # print(f'recall at {a} : {r/total:.4f} {r}/{total}')
        recall.append({'topk': a, 'recall': r / total, 'count': r})

    for a in range(10000, 110000, 10000):
        r = np.where(out < a)[0].shape[0]
        # print(f'recall at {a} : {r/total:.4f} {r}/{total}')
        recall.append({'topk': a, 'recall': r / total, 'count': r})

    recall = pd.DataFrame(recall)
    recall.to_csv(recall_csv, index=False)

    hist = pd.DataFrame(np.bincount(out))
    print(hist)
    hist.to_csv(histogram)


if __name__ == '__main__':
    res_ele = [(16, 1024), (16, 512), (16, 256), (16, 128), (32, 512), (32, 256), (32, 128), (64, 256), (64, 128)]
    ele = [(32, 625)]
    resnet = [(10,2048),(16, 1250), (32, 625)]

    for element in ele:
        path = f'/mldisk/nfs_shared_/my_vlad/test2/vlad/resnet/k{element[0]}_{element[1]}'  # 기존 파일(f1score계산용) 경로
        feature_path = f'/mldisk/nfs_shared_/my_vlad/test2/vlad/resnet/k{element[0]}_{element[1]}_for_rank'  # 평균랭킹 구하기용.
        make_directory(path, feature_path)
        name = f'resnet_rmac_vlad_k{element[0]}_{element[1]}'     # pkl, csv 저장할 파일 이름
        get_result1(name, feature_path)  # pkl 파일 저장
        get_result2(name)               # recall, history 계산 csv 파일 저장

    path = f'/mldisk/nfs_shared_/my_vlad/test2/bow/resnet_local_k20000'  # 기존 파일(f1score계산용) 경로
    feature_path = f'/mldisk/nfs_shared_/my_vlad/test2/bow/resnet_local_k20000_for_ranking'  # 평균랭킹 구하기용.
    make_directory(path, feature_path)
    name = f'resnet_rmac_bow_k20000'  # pkl, csv 저장할 파일 이름
    get_result1(name, feature_path)  # pkl 파일 저장
    get_result2(name)