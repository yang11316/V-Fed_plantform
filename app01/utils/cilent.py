from model import restnet,mlp,kddcup99,secom
from model.load_data import load_data
import torch
class Cilent(object):
    def __init__(self,conf,id=-1):
        self.conf=conf
        self.device_id = id
        if self.conf.model_type=='restnet18':
            self.local_model= restnet.ResNet18().to(conf.device)
        elif self.conf.model_type=='mlp':
            self.local_model= mlp.MLP().to(conf.device)
        elif self.conf.model_type == 'kddcup99':
            self.local_model = kddcup99.KDD().to(conf.device)
        elif self.conf.model_type=='secom':
            self.local_model=secom.SECOM().to(conf.device)
        else:
            self.local_model = mlp.MLP().to(conf.device)
        self.train_data= load_data(conf=conf,data_type=self.conf.data_type,batchsize=self.conf.batch_size,device_id=self.device_id,clients=self.conf.clients)


    def local_train(self,global_model):
        # 客户端首先用服务器端下发的全局模型覆盖本地模型
        for name , param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        #本地训练模型
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001)
        for epoch in range(self.conf.local_epochs):
            self.local_model.train()
            for batchidx, (x, label) in enumerate(self.train_data):
                if self.conf.data_type=='mnist':
                    x=x.view(-1, 28*28)
                x, label = x.to(self.conf.device), label.to(self.conf.device)
                optimizer.zero_grad()
                logit=self.local_model(x)
                loss = torch.nn.functional.cross_entropy(logit, label)
                loss.backward()
                optimizer.step()
            print('client', self.device_id, ': epoch ', epoch, 'loss:', loss.item())
        #计算本地与全局模型差
        model_difference=dict()
        for name, data in self.local_model.state_dict().items():
            model_difference[name] = (data - global_model.state_dict()[name])
        return model_difference,self.device_id,loss.item()

    def done(self):
        torch.cuda.empty_cache()
