import os
import django
from celery import Celery
from django.conf import settings

# 设置系统环境变量，安装django，必须设置，否则在启动celery时会报错
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VFed_plantform.settings')
app=Celery('VFed_plantform')
app.config_from_object('django.conf:settings',namespace='CELERY')
#去每个已经注册的app中读取文件
app.autodiscover_tasks()