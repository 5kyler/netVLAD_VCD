from VCD.utils.load import *
from VCD.models.pooling import *
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA


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


## 1-3. vlad 를 수행하는 코드
### x (M x D matrix) : 한 프레임에 대한 디스크립터
### dictionary : kmeans clustering으로 계산된 딕셔너리
def getVLAD(x, dictionary):
    predicted = dictionary.predict([x])  # x의 각 element 중에서 가장 가까운 군집을 예측함
    clusters = dictionary.cluster_centers_  # clustering 후 생성된 clusters
    k = dictionary.n_clusters  # 설정한 클러 스터수
    x = np.asmatrix(x)
    eps = 1e-6
    m, d = x.shape
    vlad_vector = np.zeros([k, d], dtype='float32')

    for i in range(k):
        if np.sum(predicted == i) > 0:
            vlad_vector[i] = np.sum(x[predicted == i, :] - clusters[i], axis=0)

    vlad_vector = vlad_vector.flatten()     # (1, k*d) 형태로 바꾸기
    vlad_vector = np.sign(vlad_vector) * np.sqrt(np.abs(vlad_vector))   # square_root normalize
    vlad_vector = vlad_vector / (np.sqrt(np.dot(vlad_vector, vlad_vector)) + eps)   # L2 normalize

    return vlad_vector


## 1-4. vlad를 pth 파일로 저장하는 코드
### path : 피쳐가 저장되어있는 경로
### dictionary : kmeansclustering으로 계산된 딕셔너리
def saveVLADVectorPth(path, dictionary):
    PATH = '/mldisk/nfs_shared_/my_vlad/test2/vlad_feature/k16_400'      #vlad 피쳐 저장할 경로
    os.makedirs(PATH)
    for folder in os.listdir(path):
        if not os.path.isdir(os.path.join(PATH, folder)):
            os.mkdir(os.path.join(PATH, folder))
        for featurePath in os.listdir(os.path.join(path, folder)):
            feature, m = load_file(os.path.join(path, folder, featurePath))
            vlad_list = []
            for row in range(0, len(feature)):
                vlad = getVLAD(feature[row], dictionary)
                vlad_list.append([vlad])
            vlad_features = np.vstack(vlad_list)
            vlad_features = torch.tensor(vlad_features, dtype=torch.float32)

            torch.save(vlad_features, PATH + '/' + folder + '/' +featurePath)


if __name__ == '__main__':
    # path = '/mldisk/nfs_shared_/my_vlad/resnet_features/core_dataset' # resnet 피쳐 저장경로
    # path = '/mldisk/nfs_shared_/my_vlad/resnet_segment_features'  # resnet segemnt 경로

    # path = "/mldisk/nfs_shared_/MLVD/VCDB-core/features/mobilenet-avg-pretrained/frame-features/core_dataset"   # mobilenet 피쳐 저장경로
    # path = '/mldisk/nfs_shared_/MLVD/VCDB-core/features/mobilenet-avg-pretrained/segment-maxpool-5/core_dataset'  # segment feature 저장 경로
    # path = '/mldisk/nfs_shared_/my_vlad/mobile_avg/VCDB-core/k16/core_dataset_vlad'
    path = '/mldisk/nfs_shared_/my_vlad/test2/pca/420' # feature 저장 경로
    k = 16  # 사용할 클러스터수
    descriptor_set = getDescriptorSet(path)
    dictionary = getDictionary(descriptor_set, k)
    saveVLADVectorPth(path, dictionary)


