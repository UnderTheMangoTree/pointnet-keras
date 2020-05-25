import numpy as np
import h5py
import os
#最远点采样
def farthest_point_sample(xyz, npoint):
    N, C = xyz.shape
    centroids = np.zeros(npoint)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)# random select one as centroid
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].reshape(1, 3)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)# select the farthest one as centroid
        #print('index:%d, dis:%.3f'%(farthest,np.max(distance)))
    return centroids
#读取off文件
def read_off(filename):
    npoints=2048
    f = open(filename)
    f.readline()  # ignore the 'OFF' at the first line
    f.readline()  # ignore the second line
    All_points = []
    selected_points = []
    while True:
        new_line = f.readline()
        x = new_line.split(' ')
        if x[0] != '3':
            A = np.array(x[0:3], dtype='float32')
            All_points.append(A)
        else:
            break
    # if the numbers of points are less than 2000, extent the point set
    if len(All_points) < (npoints + 3):
        print("none")
        return None
    All_points = np.array(All_points)
    # take and shuffle points
    # index = np.random.choice(len(All_points), num_select, replace=False)
    xyz = All_points.copy()
    points_farthest_index = farthest_point_sample(xyz, npoints).astype(np.int64)
    points_farthest = xyz[points_farthest_index, :]
    centroid = np.mean(points_farthest, axis=0)
    points_unit_sphere = points_farthest - centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points_farthest) ** 2, axis=-1)))
    points_unit_sphere /= furthest_distance
    return list(points_unit_sphere)  # return N*3 array的list
#保存h5
def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()

cur_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(cur_path, "ModelNet40")
# list of all the categories
directories = [d for d in os.listdir(dir_path)
               if os.path.isdir(os.path.join(dir_path, d))]

load_dict = [["train", "ModelNet40/train"], ["test", "ModelNet40/test"]]
flag="train_labels_name"
for d in load_dict:
    label_names = {}
    for i in range(len(directories)):
        label_name = directories[i]
        train_path = os.path.join(dir_path, directories[i], d[0])
        save_path = os.path.join(cur_path, d[1])
        All_points = None
        label = []
        label_names[i]=label_name
        # all the files in "train" floder
        for filename in os.listdir(train_path):
            print('read:',filename)
            if '.off' in filename:
                s = os.path.join(train_path, filename)
                points = read_off(s)

                if All_points is None:
                    if points:

                        All_points = points
                        label.append(i)
                elif points:
                    All_points = np.vstack((All_points, points))
                    label.append(i)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data_save_path = os.path.join(save_path, directories[i] + '.h5')
        save_h5(data_save_path, All_points, label)
        try:
            print(All_points.shape)
            print(len(label))
        except:
            print(type(All_points))
            print("eer")
    print(label_names)
    import pickle
    output = open('%s.pkl'%flag, 'wb')
    #保存label的name
    pickle.dump(label_names, output)
    output.close()
    flag="test_labels_name"

##################将40个类别的h5整合成一个################################

data_path = 'ModelNet40/'
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)
for d in [['train', len(os.listdir(data_path + 'train'))], ['test', len(os.listdir(data_path + 'test'))]]:
    data = None
    labels = None
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, data_path+d[0])
    filenames = [d for d in os.listdir(path)]
    for s in filenames:
        cur_data,cur_labels=load_h5(os.path.join(path, s))
        cur_data = cur_data.reshape(1, -1, 3)
        cur_labels = cur_labels.reshape(1, -1)
        if labels is None or data is None:
            labels=cur_labels
            data=cur_data
        else:
            labels = np.hstack((labels, cur_labels))
            data = np.hstack((data, cur_data))

    data = data.reshape(-1, 2048, 3)
    labels = labels.reshape(-1, 1)
    save_name = data_path + '/ply_data_{0}.h5'.format(d[0])
    print(data.shape)
    print(labels.shape)
    h5_fout = h5py.File(save_name)
    h5_fout.create_dataset(
        'data', data=data,
        dtype='float32')
    h5_fout.create_dataset(
        'label', data=labels,
        dtype='float32')
    h5_fout.close()
