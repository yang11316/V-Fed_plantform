from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):

        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = F.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)
        # # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after conv:', x.shape) #[b, 512, 2, 2]
        # [b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x





# class Restnet_run():
#     def __init__(self,batchsize,data_type,local_epochs=5,device='cuda',device_id='-1',clients=5):
#         self.batchsize=batchsize
#         self.local_epochs=local_epochs
#         self.data_type=data_type
#         self.device=torch.device(device)
#         self.model=ResNet18().to(self.device)
#         #使用交叉熵
#         self.criteon = nn.CrossEntropyLoss().to(self.device)
#         #使用Adam优化器
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
#         self.device_id=device_id
#         self.clients=clients
#     def run(self):
#         train_data= load_data(data_type=self.data_type,batchsize=self.batchsize,device_id=self.device_id,clients=self.clients)
#         for epoch in range(self.local_epochs):
#             self.model.train()
#             for batchidx, (x, label) in enumerate(train_data):
#                 x, label = x.to(self.device), label.to(self.device)
#                 logits = self.model(x)
#                 # logits: [b, 10]
#                 # label:  [b]
#                 # loss: tensor scalar
#                 loss = self.criteon(logits, label)
#                 # backprop
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#             print( 'client',self.device_id,': epoch ', epoch, 'loss:', loss.item())
#     def eval(self):
#         eval_data=load_data(data_type=self.data_type,batchsize=self.batchsize,device_id=self.device_id,clients=self.clients)
#         self.model.eval()
#         with torch.no_grad():
#             # test
#             total_loss=0.0
#             total_correct = 0
#             total_num = 0
#             for x, label in eval_data:
#                 x, label = x.to(self.device), label.to(self.device)
#                 logits = self.model(x)
#                 pred = logits.argmax(dim=1)
#                 correct = torch.eq(pred, label).float().sum().item()
#                 total_correct += correct
#                 total_num += x.size(0)
#                 self.criteon.reduction='sum'
#                 total_loss+=self.criteon(logits, label).item()
#             acc = total_correct / float(total_num)
#             total_loss=total_loss/total_num
#             # print('test acc:', acc)
#             # print('test loss:',total_loss)
#             return acc,total_loss
#     def done(self):
#         torch.cuda.empty_cache()
#     def getmodel(self):
#         return self.model

