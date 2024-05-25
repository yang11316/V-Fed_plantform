from celery import shared_task

from app01 import models
from app01.utils.flcontrol import *


@shared_task
def flrun(taskid):
    print(taskid, "running")
    # 读取任务信息
    taskinfo = models.TaskInfo.objects.filter(taskid=taskid).values()[0]
    # 传值给联邦学习控制函数
    if taskinfo['algorithm']=='VFed-avg':
        data,svote,cvote = start_fl(taskinfo)
        # 保存任务数据
        models.Taskdata.objects.create(taskid=taskid, data=data,svote=svote,cvote=cvote)
    elif taskinfo['algorithm']=='Fed-avg':
        data = start_fl(taskinfo)
        models.Taskdata.objects.create(taskid=taskid,data=data)
    # 任务完成，修改任务状态为结束状态
    models.TaskInfo.objects.filter(taskid=taskid).update(status='2')
