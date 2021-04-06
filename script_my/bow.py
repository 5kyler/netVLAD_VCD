from VCD.utils.load import *
from VCD.models.pooling import *
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
import scipy.cluster.vq as vq

###############################################################################################################################################

# 1. descriptor 를 vlad vector로 변환
## 1-1. descriptor 를 set(list)으로 만들기
### path : 디스크립터들이 저장된 폴더 경로  /workspace/features/VCDB/core_dataset
def getDescriptorSet(path):
    descriptors = []
    for folder in os.listdir(path):
        for descriptorPath in os.listdir(os.path.join(path, folder)):
            descriptor, _ = load_file(os.path.join(path, folder, descriptorPath))
            descriptors.append(descriptor)
    descriptor_set = np.vstack(descriptors)
    print("descriptor_set 생성 완료")
    print(descriptor_set.shape)
    return descriptor_set


## 1-2. get dictionary (KMeans)
### descriptor_set : 추출된 디스크립터들 하나의 set로 묶음
### k : 클러스터링 하려는 갯수
def getDictionary(descriptor_set, k):
    print('kmeans clustering시작')
    dictionary = MiniBatchKMeans(n_clusters=k, random_state=0).fit(descriptor_set.astype('double'))
    print('kmeans clustering 완료')
    return dictionary


def encoding_features(path):
    codebook = np.load('/mldisk/nfs_shared_/js/codebook/resnet50_1fps/k20000_codebook_resnet50.npy', allow_pickle=True)
    PATH = '/mldisk/nfs_shared_/my_vlad/test2/bow/resnet_local_k20000' # encode feature 저장할 path

    for folder in os.listdir(path):
        if not os.path.isdir(os.path.join(PATH, folder)):
            os.mkdir(os.path.join(PATH, folder))
        for featurePath in os.listdir(os.path.join(path, folder)):
            feature, m = load_file(os.path.join(path, folder, featurePath))
            bow_list = []
            for row in range(0, len(feature)):
                encode_feature = np.zeros((20000,), "float32")
                descriptor = np.expand_dims(feature[row], axis=0)

                code, dist = vq.vq(descriptor, codebook)
                for c in code:
                    encode_feature[c] += 1
                bow_list.append([encode_feature])
                bow_features = np.vstack(bow_list)
                bow_features = torch.tensor(bow_features, dtype=torch.float32)

                torch.save(bow_features, PATH + '/' + folder + '/' + featurePath)


if __name__ == '__main__':
    # path = '/mldisk/nfs_shared_/my_vlad/resnet_segment_features'
    # merging_desc = getDescriptorSet(path)
    # getDictionary(merging_desc, 20000)
    encoding_features('/mldisk/nfs_shared_/my_vlad/resnet_features/core_dataset')