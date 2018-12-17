import numpy as np
import h5py
import os
import sys

pre_s1_batch = None
pre_s2_batch = None
pre_label_batch = None
pre_step = -1
presize = 10000
global_step = 0

#从磁盘加载新数据到Buffer
def update_pre_data():
    global pre_step
    global pre_s1_batch
    global pre_s2_batch
    global pre_label_batch
    pre_step += 1

    # data_dir = "C:/zju/dataset/tianchi/training.h5"
    data_dir = "D:/zju/dataset/tianchi/training.h5"
    file = h5py.File(data_dir)
    s1_data = file['sen1']
    s2_data = file['sen2']
    label_data = file['label']
    n_samples = s1_data.shape[0]

    start_pos = pre_step * presize
    end_pos = min(start_pos + presize, n_samples)

    pre_s1_batch = np.asarray(s1_data[start_pos:end_pos, :, :, :])
    pre_s2_batch = np.asarray(s2_data[start_pos:end_pos, :, :, :])
    pre_label_batch = np.asarray(label_data[start_pos:end_pos, :])

#从Buffer中读batch
def pre_loaddata(times,step,batch_size):
    if(step/times > pre_step or pre_s1_batch is None):
        update_pre_data()

    step = step % times
    n_samples = len(pre_s1_batch)
    start_pos = step * batch_size
    end_pos = min(start_pos + batch_size,n_samples)

    s1_batch = pre_s1_batch[start_pos:end_pos,:,:,:]
    s2_batch = pre_s2_batch[start_pos:end_pos,:,:,:]
    label_batch = pre_label_batch[start_pos:end_pos,:]

    return (s1_batch,s2_batch,label_batch)

# def get_batch(step, batch_size, mode = "validate"):
#     #训练集数据从buffer中读取
#     if mode == "train":
#         times = int(presize / batch_size)
#         s1_batch,s2_batch,label_batch = pre_loaddata(times,step,batch_size)
#         x_batch = np.concatenate([s1_batch, s2_batch], 3)
#     else:
#         base_dir = "C:/zju/dataset/tianchi"
#         if  mode == "validate":
#             data_dir = os.path.join(base_dir,'validation.h5')
#         else:
#             data_dir = os.path.join(base_dir, 'round1_test_a_20181109.h5')
#         file = h5py.File(data_dir)
#         s1_data = file['sen1']
#         s2_data = file['sen2']
#         if mode != "test":
#             label_data = file['label']
#
#         n_samples = s1_data.shape[0]
#
#         start_pos = step * batch_size
#         end_pos = min(start_pos + batch_size,n_samples)
#
#         s1_batch = np.asarray(s1_data[start_pos:end_pos,:,:,:])
#         s2_batch = np.asarray(s2_data[start_pos:end_pos, :, :, :])
#         if mode != "test":
#             label_batch = np.asarray(label_data[start_pos:end_pos, :])
#         else:
#             label_batch = None
#         # print("s1_batch: %.1f" % sys.getsizeof(s1_batch))
#         # print("s2_batch: %.1f" % sys.getsizeof(s2_batch))
#         # print("label_batch: %.1f" % sys.getsizeof(label_batch))
#
#         x_batch = np.concatenate([s1_batch,s2_batch],3)
#
#
#     return (x_batch,label_batch)

def get_batch(step, batch_size, mode = "validate"):
    base_dir = "C:/zju/dataset/tianchi"
    if mode == "train":
        data_dir = os.path.join(base_dir, 'training.h5')
    elif mode == "validate":
        data_dir = os.path.join(base_dir,'validation.h5')
    else:
        data_dir = os.path.join(base_dir, 'round1_test_a_20181109.h5')
    file = h5py.File(data_dir)
    s1_data = file['sen1']
    s2_data = file['sen2']
    if mode != "test":
        label_data = file['label']

    n_samples = s1_data.shape[0]

    start_pos = step * batch_size
    end_pos = min(start_pos + batch_size,n_samples)

    s1_batch = np.asarray(s1_data[start_pos:end_pos,:,:,:])
    s2_batch = np.asarray(s2_data[start_pos:end_pos, :, :, :])
    if mode != "test":
        label_batch = np.asarray(label_data[start_pos:end_pos, :])
    else:
        label_batch = None
    # print("s1_batch: %.1f" % sys.getsizeof(s1_batch))
    # print("s2_batch: %.1f" % sys.getsizeof(s2_batch))
    # print("label_batch: %.1f" % sys.getsizeof(label_batch))

    x_batch = np.concatenate([s1_batch,s2_batch],3)


    return (x_batch,label_batch)

def save_output(y_out):
    with open("C:/zju/dataset/tianchi/output.csv",'a') as file:
        for i in y_out:
            max_pos = i.index(max(i))
            length = len(i)
            for idx in range(length):
                if idx == max_pos:
                    file.write('1')
                else:
                    file.write('0')
                if idx == length-1:
                    file.write('\n')
                else:
                    file.write(',')


if __name__ == '__main__':
    get_batch(0,10)