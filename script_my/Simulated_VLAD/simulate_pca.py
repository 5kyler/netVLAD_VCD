from sklearn.decomposition import IncrementalPCA

import torch
import numpy as np
import os
from VCD.utils import *


def getDescriptorSet(path):
    descriptors = []
    for folder in os.listdir(path):
        if folder == 'original':
            for descriptorPath in os.listdir(os.path.join(path, folder)):
                descriptor, _ = load_file(os.path.join(path, folder, descriptorPath))
                descriptors.append(descriptor)
        else:
            for level in os.listdir(os.path.join(path, folder)):
                for descriptorPath in os.listdir(os.path.join(path, folder, level)):
                    descriptor, _ = load_file(os.path.join(path, folder, level, descriptorPath))
                    descriptors.append(descriptor)
    descriptor_set = np.vstack(descriptors)
    print("descriptor_set 생성 완료")
    print(descriptor_set.shape)
    return descriptor_set


def featurePCA(path):
    PATH = '/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/400/sim'    #pca 저장할경로

    tmp = 0
    feat = getDescriptorSet(path)

    print("pca 시작 제발 메모리 터지지말아주세요")
    n_batches = 20
    inc_pca = IncrementalPCA(n_components=400, whiten=True)

    for bat in np.array_split(feat, n_batches):
        print(".", end="")
        inc_pca.partial_fit(bat)

    pca_matrix = inc_pca.transform(feat)
    print("pca완료 이제 차곡차곡 나눠서 저장하자")
    for folder in os.listdir(path):
        if not os.path.isdir(os.path.join(PATH, folder)):
            os.mkdir(os.path.join(PATH, folder))
        if folder == 'original' :
            for featurePath in os.listdir(os.path.join(path, folder)):
                feature, m = load_file(os.path.join(path, folder, featurePath))
                pca_list = []
                for i in range(tmp, tmp+m):
                    pca_list.append(pca_matrix[i])
                pca_feat = np.vstack(pca_list)
                pca_feat = torch.tensor(pca_feat, dtype=torch.float32)
                torch.save(pca_feat, PATH + '/' + folder + '/' + featurePath)
                tmp = tmp+m
        else:
            for level in os.listdir(os.path.join(path, folder)):
                if not os.path.isdir(os.path.join(PATH, folder, level)):
                    os.mkdir(os.path.join(PATH, folder, level))
                for featurePath in os.listdir(os.path.join(path, folder, level)):
                    feature, m = load_file(os.path.join(path, folder, level, featurePath))
                    pca_list = []
                    for i in range(tmp, tmp + m):
                        pca_list.append(pca_matrix[i])
                    pca_feat = np.vstack(pca_list)
                    pca_feat = torch.tensor(pca_feat, dtype=torch.float32)
                    torch.save(pca_feat, PATH + '/' + folder + '/' + level + '/' + featurePath)
                    tmp = tmp + m
    print("하나님 감사합니다!")


if __name__ == '__main__':

    path = '/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/dataset/resnet_rmac_simulated_dataset'   # resnet segemnt 경로

    featurePCA(path)
