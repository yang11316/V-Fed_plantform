import torch

from model import restnet, mlp,kddcup99,secom
from model.load_data import load_data




class Server(object):
    def __init__(self, conf, device_id):
        self.conf = conf
        self.device_id = device_id
        self.diff_dic = {}  # 保存clients的diff
        self.clientloss_dic={}#保存client的loss，用于计算
        self.loss_dic = {}  # 保存server计算的loss
        self.server_vote = {}  # 模型聚合的值
        self.client_vote={}
        self.la = {}  # 模型聚合的权重

        for i in range(1, conf.clients + 1):
            self.server_vote[i] = 1
            self.client_vote[i]=1
        if self.conf.model_type == 'restnet18':
            # server的id为0,id和取数据集相关联
            self.server_model = restnet.ResNet18().to(conf.device)

        elif self.conf.model_type == 'mlp':
            self.server_model = mlp.MLP().to(conf.device)
        elif self.conf.model_type == 'kddcup99':
            self.server_model = kddcup99.KDD().to(conf.device)
        elif self.conf.model_type=='secom':
            self.server_model=secom.SECOM().to(conf.device)
        else:
            self.server_model = kddcup99.KDD().to(conf.device)
        self.test_data, self.eval_data = load_data(conf=conf,data_type=self.conf.data_type, batchsize=self.conf.batch_size,
                                                   device_id=self.device_id,
                                                   clients=self.conf.clients)

    # 模型聚合
    def model_aggregate(self, weight_accumulator, la):

        model_dict = self.server_model.state_dict()
        for name, data in self.server_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * la
            if data.type() != update_per_layer.type():
                model_dict[name].add_(update_per_layer.to(torch.int64))
            else:
                model_dict[name].add_(update_per_layer)
        self.server_model.load_state_dict(model_dict)

    # 计算acc和loss
    def model_eval(self):

        self.server_model.eval()
        with torch.no_grad():
            # test
            total_loss = 0.0
            total_correct = 0
            total_num = 0
            for x, label in self.test_data:
                x, label = x.to(self.conf.device), label.to(self.conf.device)
                if self.conf.data_type == 'mnist':
                    x = x.view(-1, 28 * 28)
                logits = self.server_model(x)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)

                total_loss += torch.nn.functional.cross_entropy(logits, label,
                                                                reduction='sum').item()
            acc = total_correct / float(total_num)
            total_loss = total_loss / total_num
            return acc, total_loss

    # 模型最终聚合
    def model_aggregate_sum(self):
        #计算融合权重
        weight1=0.3
        weight2=0.7

        sum_server_vote = sum(self.server_vote.values())
        sum_client_vote = sum(self.client_vote.values())
        for i in range(1, self.conf.clients + 1):
            self.la[i] = float((self.server_vote[i] / sum_server_vote)*weight1)+float((self.client_vote[i]/sum_client_vote)*weight2)
        for i in range(1, self.conf.clients+1):
            self.model_aggregate(self.diff_dic[i], self.la[i])

    # 取最小的前k个，weight加一
    def server_vote_cal(self):
        loss_k = dict(
            sorted(self.loss_dic.items(), key=lambda x: x[1], reverse=False)[:self.conf.k]).keys()  # 获取loss最小的前k个的key值
        print(loss_k)
        for i in loss_k:
            self.server_vote[i] += 1
        print("server vote:",self.server_vote)
        #计算la，模型聚合
        # sum_weight = sum(self.server_vote.values())
        # for i in range(1, self.conf.clients + 1):
        #     self.la[i] = float(self.server_vote[i] / sum_weight)

    # 计算每个clients的loss
    def loss_cal(self):

        torch.save(self.server_model.state_dict(), "./model/tmp.pt")
        for i in range(1, self.conf.clients + 1):
            with torch.no_grad():
                self.model_aggregate(self.diff_dic[i], 1)  # diff暂时更新到model中
                self.server_model.eval()
                total_loss = 0.0
                total_num = 0
                for x, label in self.test_data:
                    x, label = x.to(self.conf.device), label.to(self.conf.device)
                    if self.conf.data_type == 'mnist':
                        x = x.view(-1, 28 * 28)
                    logits = self.server_model(x)
                    total_num += x.size(0)
                    total_loss += torch.nn.functional.cross_entropy(logits, label,
                                                                    reduction='sum').item()
                total_loss = total_loss / total_num
                self.loss_dic[i] = total_loss
                self.server_model.load_state_dict(torch.load("./model/tmp.pt"))

    #排列投票每个client的loss，取最小的前k个
    def client_vote_cal(self):
        loss_k = dict(
            sorted(self.clientloss_dic.items(), key=lambda x: x[1], reverse=False)[:self.conf.k]).keys()  # 获取loss最小的前k个的key值
        for i in loss_k:
            self.client_vote[i]+=1
        print("client vote:",self.client_vote)



    # server收到diff和id并进行暂时存储、更新
    def recept_diff(self, diff, id,client_loss):
        self.clientloss_dic[id]=client_loss
        self.diff_dic[id] = diff

    # 清除cuda内存
    def done(self):
        torch.cuda.empty_cache()

    # 计算攻击率
    def attack_rate(self):
        self.server_model.eval()
        with torch.no_grad():
            label_dic={1:7}
            attack_rate=[]
            for label_key in label_dic.keys():
                count_from_to = 0
                count_from = 0
                for x, label in self.test_data:
                    x, label = x.to(self.conf.device), label.to(self.conf.device)
                    if self.conf.data_type == 'mnist':
                        x = x.view(-1, 28 * 28)
                    logits = self.server_model(x)
                    pred = logits.argmax(dim=1)
                    count_from += int((label == label_key).sum())
                    # index=torch.nonzero(label==1).squeeze()
                    index = [index for index, value in enumerate(label) if value == label_key]
                    for i in index:
                        if pred[i] == label_dic[label_key]:
                            count_from_to += 1
                attack_rate.append(100 * (count_from_to / count_from))
            return float(sum(attack_rate)/len(attack_rate))

    # 计算eval_data上的准确率和loss
    def model_eval_data(self):
        self.server_model.eval()
        with torch.no_grad():
            # test
            total_loss = 0.0
            total_correct = 0
            total_num = 0
            for x, label in self.eval_data:
                x, label = x.to(self.conf.device), label.to(self.conf.device)
                if self.conf.data_type == 'mnist':
                    x = x.view(-1, 28 * 28)
                logits = self.server_model(x)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)

                total_loss += torch.nn.functional.cross_entropy(logits, label,
                                                                reduction='sum').item()
            acc = total_correct / float(total_num)
            total_loss = total_loss / total_num
            return acc, total_loss