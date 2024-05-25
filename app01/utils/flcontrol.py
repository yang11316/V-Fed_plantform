
from .server import *
from .cilent import *
import random
# 定义一个配置类
class Conf():
    def __init__(self, global_epochs, local_epochs, data_type, model_type, k, clients):
        self.global_epochs = global_epochs
        self.local_epochs = local_epochs
        self.batch_size = 50
        self.data_type = data_type  # cifar,mnist,kddcup99,secom

        if model_type == 'default':
            sw = {"cifar": "restnet18", "mnist": "mlp", "kddcup99": "kddcup99", "secom": "secom"}
            self.model_type = sw.get(data_type, "nothing")
        else:
            self.model_type=model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.la = round(1/k,2)
        self.momentum = 0.0001
        self.k = k  # 一次从clients抽取几个
        self.clients = clients  # 10,5
        self.data_test = '/'  # '_A5/'


def start_fl(taskinfo):
    taskid=taskinfo["taskid"]
    global_epochs=taskinfo["global_iteration"]
    local_epochs=taskinfo["local_iteration"]
    algorithm=taskinfo["algorithm"]
    client_num=taskinfo["client_num"]
    k=taskinfo["chose_num"]
    data_type=taskinfo["data_type"]
    model_type=taskinfo["deeplg_alg"]

    # 实例化配置类
    conf=Conf(global_epochs=global_epochs,local_epochs=local_epochs,data_type=data_type,model_type=model_type,k=k,clients=client_num)
    # 首先选择联邦学习算法
    if algorithm=="VFed-avg":

        #实例化server、client类
        server = Server(conf, device_id=0)
        clients = []
        for i in range(1, conf.clients + 1):
            clients.append(Cilent(conf, id=i))
        # 预训练
        print("pretraining!")
        weight_accumulator1 = {}
        for name, params in server.server_model.state_dict().items():
            # 生成一个空的weight张量
            weight_accumulator1[name] = torch.zeros_like(params)
        for c in clients:
            diff, c_id, c_loss = c.local_train(server.server_model)
            for name, params in server.server_model.state_dict().items():
                # 把每个client都加一块
                weight_accumulator1[name].add_(diff[name])
        # weight聚合
        server.model_aggregate(weight_accumulator1, la=round(1 / conf.clients,2))

        # training
        data={}
        for global_epoch in range(conf.global_epochs):
            # client本地训练
            for c in clients:
                diff, c_id, c_loss = c.local_train(server.server_model)
                server.recept_diff(diff, c_id, c_loss)
            # weight聚合
            server.loss_cal()
            server.server_vote_cal()
            server.client_vote_cal()
            server.model_aggregate_sum()

            # print(list(server.server_model.model.state_dict().items())[3])
            acc, loss = server.model_eval()
            print("global Epoch %d, acc: %f, loss: %f" % (global_epoch, acc, loss))
            acc1, loss1 = server.model_eval_data()
            print("model eval data1, acc: %f, loss: %f" % (acc1, loss1))
            attack_rate = server.attack_rate()
            print("attack_rate: %f" % (attack_rate))
            #保存数据，返回前端展示
            data[str(global_epoch+1)]=[acc,loss,attack_rate]

        torch.save(server.server_model.state_dict(), "./taskmod/"+str(taskid)+".pt")
        svote=server.server_vote
        cvote=server.client_vote
        server.done()
        for c in clients:
            c.done()
        return data,svote,cvote

    elif algorithm =="Fed-avg":
        # 初始化server和client
        server = Server(conf, device_id=0)
        clients = []
        for i in range(1, conf.clients + 1):
            clients.append(Cilent(conf, id=i))

        # 预训练
        weight_accumulator1 = {}
        for name, params in server.server_model.state_dict().items():
            # 生成一个空的weight张量
            weight_accumulator1[name] = torch.zeros_like(params)
        for c in clients:
            diff ,c_id, c_loss= c.local_train(server.server_model)
            for name, params in server.server_model.state_dict().items():
                # 把每个client都加一块
                weight_accumulator1[name].add_(diff[name])


        # weight聚合
        server.model_aggregate(weight_accumulator1,la=round(1 / conf.clients,2))
        data = {}

        for global_epoch in range(conf.global_epochs):
            weight_accumulator = {}
            # 每次从client里面随机选k个
            chose_clients = random.sample(clients, conf.k)
            for name, params in server.server_model.state_dict().items():
                # 生成一个空的weight张量
                weight_accumulator[name] = torch.zeros_like(params)

            # client本地训练
            for c in chose_clients:
                diff,c_id, c_loss = c.local_train(server.server_model)
                for name, params in server.server_model.state_dict().items():
                    # 把每个client都加一块
                    weight_accumulator[name].add_(diff[name])
            # weight聚合

            server.model_aggregate(weight_accumulator, la=conf.la)
            # print(list(server.server_model.model.state_dict().items())[3])
            acc, loss = server.model_eval()
            print("global Epoch %d, acc: %f, loss: %f\n" % (global_epoch, acc, loss))

            acc1, loss1 = server.model_eval_data()
            print("model eval data1, acc: %f, loss: %f" % (acc1, loss1))
            attack_rate = server.attack_rate()
            print("attack_rate:%f" % attack_rate)
            data[str(global_epoch + 1)] = [acc, loss, attack_rate]
        torch.save(server.server_model.state_dict(), "./taskmod/" + str(taskid) + ".pt")
        server.done()
        for c in clients:
            c.done()
        return data