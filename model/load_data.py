import torch
from torch.utils.data import DataLoader

def load_data(conf,data_type, batchsize, device_id, clients, shuffle=True, num_workers=0, drop_last=False):
    # 使用DataLoader加载数据并返回
    test=conf.data_test
    if device_id == 0:
        test_path = "./data/" + data_type + '_split_' + str(clients) + test + data_type + '_test_data'
        eval_path = "./data/" + data_type + '_split_' + str(clients) + test + data_type + '_eval_data'
        test = torch.load(test_path)
        eval = torch.load(eval_path)
        test_data = DataLoader(dataset=test, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers,
                               drop_last=drop_last)
        eval_data = DataLoader(dataset=eval, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers,
                               drop_last=drop_last)
        return test_data, eval_data
    else:
        data_path = "./data/" + data_type + '_split_' + str(clients) + test + data_type + '_data' + str(device_id)
        data = torch.load(data_path)
        data = DataLoader(dataset=data, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers,
                          drop_last=drop_last)
        return data
    # if data_type=='cifar':
    #
    #     if clients == 5:
    #         if device_id==0:
    #             data_path="./data1/cifar_split5/cifar_test_data"
    #         else:
    #             data_path="./data1/cifar_split5/cifar_data"+str(device_id)
    #     elif clients==10:
    #         if device_id==0:
    #             data_path="./data1/cifar_split10/cifar_test_data"
    #         else:
    #             data_path="./data1/cifar_split10/cifar_data"+str(device_id)
    # elif data_type=='mnist':
    #
    #     if clients == 5:
    #         if device_id == 0:
    #             data_path = "./data1/mnist_split5/mnist_test_data"
    #         else:
    #             data_path = "./data1/mnist_split5/mnist_data" + str(device_id)
    #     elif clients == 10:
    #         if device_id == 0:
    #             data_path = "./data1/mnist_split10/mnist_test_data"
    #         else:
    #             data_path = "./data1/mnist_split10/mnist_data" + str(device_id)
    #
    # else:
    #     data_path=''
