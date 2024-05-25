from django.db import models

# Create your models here.

class UserLogin(models.Model):
    """" 用户信息表  """
    username=models.CharField(verbose_name='用户名称',max_length=32)
    userid=models.CharField(verbose_name='用户id',max_length=32)
    password=models.CharField(verbose_name='用户密码',max_length=32)
    uemail=models.EmailField(verbose_name='注册邮箱',null=False, blank=False)


class TaskInfo(models.Model):
    """任务信息表"""
    taskname=models.CharField(verbose_name='任务名称',max_length=32)
    taskid=models.CharField(verbose_name='任务id',max_length=32)
    description=models.TextField(verbose_name='数据集描述')
    global_iteration=models.IntegerField(verbose_name='全局迭代次数')
    local_iteration=models.IntegerField(verbose_name='本地迭代次数')
    algorithm=models.CharField(verbose_name='联邦学习算法',max_length=32)
    create_time=models.DateTimeField(verbose_name='创建时间',auto_now_add=True)
    status=models.IntegerField(verbose_name='任务状态')
    # status任务状态：0->等待，1->进行中，2->结束
    client_num=models.IntegerField(verbose_name="使用节点数",default=10)
    chose_num=models.IntegerField(verbose_name="每轮选择节点数",default=5)
    data_type=models.CharField(verbose_name="数据集类型",max_length=32,default='mnist')
    deeplg_alg=models.CharField(verbose_name="深度学习算法",max_length=32,default='default')



class UserTask(models.Model):
    """  用户任务表 """
    userid=models.CharField(verbose_name='用户id',max_length=32)
    taskid=models.CharField(verbose_name='任务id',max_length=32)


class Taskdata(models.Model):
    """ 任务信息表 """
    taskid=models.CharField(verbose_name='任务id',max_length=32)
    data=models.TextField(verbose_name='数据accloss')
    svote=models.TextField(verbose_name="server_vote",default=None,null=True,blank=True)
    cvote=models.TextField(verbose_name='client_vote',default=None,null=True,blank=True)
