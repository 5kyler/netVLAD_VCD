from sklearn.decomposition import IncrementalPCA
from script_my.VLAD.vlad import getDescriptorSet
import torch
import numpy as np
import os
from VCD.utils import *


def featurePCA(path):
    PATH = '/mldisk/nfs_shared_/my_vlad/test2/pca/420'    #pca 저장할경로
    os.mkdir(PATH)
    tmp = 0
    feat = getDescriptorSet(path)
    print("pca 시작 제발 메모리 터지지말아주세요")
    n_batches = 46
    inc_pca = IncrementalPCA(n_components=420, whiten=True)

    for bat in np.array_split(feat, n_batches):
        print(".", end="")
        inc_pca.partial_fit(bat)

    pca_matrix = inc_pca.transform(feat)
    print("pca완료 이제 차곡차곡 나눠서 저장하자")
    for folder in os.listdir(path):
        if not os.path.isdir(os.path.join(PATH, folder)):
            os.mkdir(os.path.join(PATH, folder))

        for featurePath in os.listdir(os.path.join(path, folder)):
            feature, m = load_file(os.path.join(path, folder, featurePath))
            pca_list = []
            for i in range(tmp, tmp+m):
                pca_list.append(pca_matrix[i])
            pca_feat = np.vstack(pca_list)
            pca_feat = torch.tensor(pca_feat, dtype=torch.float32)
            torch.save(pca_feat, PATH + '/' + folder + '/' + featurePath)
            tmp = tmp+m
    print("하나님 감사합니다!")


if __name__ == '__main__':
    # path = '/mldisk/nfs_shared_/my_vlad/resnet_features/core_dataset' # resnet 피쳐 저장경로
    path = '/mldisk/nfs_shared_/my_vlad/resnet_segment_features'   # resnet segemnt 경로
    # path = "/mldisk/nfs_shared_/MLVD/VCDB-core/features/mobilenet-avg-pretrained/frame-features/core_dataset"   # mobilenet 피쳐 저장경로
    # path = '/mldisk/nfs_shared_/MLVD/VCDB-core/features/mobilenet-avg-pretrained/segment-maxpool-5/core_dataset'  # segment feature 저장 경로
    # path = '/mldisk/nfs_shared_/my_vlad/mobile_avg/VCDB-core/k16/core_dataset_vlad'
    featurePCA(path)
