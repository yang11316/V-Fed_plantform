import json
import re
import datetime
from django.core.mail import send_mail
from django.core.paginator import Paginator  # django分页组件
from django.shortcuts import render, HttpResponse, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, Http404, FileResponse, HttpResponseRedirect
from app01.utils.crypto import MD5_hashCode
import os

# Create your views here.

# ----------------------装饰器---------------------------

# 检测登录装饰器
def login_check(func):
    def inner(request, *args, **kwargs):
        if not request.session.get('is_login'):
            return redirect('/login/')
        else:
            return func(request, *args, **kwargs)

    return inner


# ----------------------工具类---------------------------
# 从session里面获取登录信息
def get_session_userinfo(request):
    user_info = request.session['user_info']
    username = user_info['username']
    userid = MD5_hashCode(user_info['useremail'])
    return username, userid


# 检查任务信息
def task_info_check(request, taskinfo_list):
    re_list = [' ', 'and', '#', '--', 'union', 'select', '$', 'handler', '/**/']
    taskname = taskinfo_list['taskname']
    algorithm = taskinfo_list['algorithm']
    global_epoch = taskinfo_list['global_epoch']
    local_epoch = taskinfo_list['local_epoch']
    client_num = taskinfo_list['client_num']
    chose_num = taskinfo_list['chose_num']
    for i in re_list:
        if i in taskname or i in algorithm:
            return HttpResponse("<script>alert('包含敏感字符，请从新输入');history.back(-1)</script>")
    if not global_epoch.isdigit():
        return render(request, 'error.html', {"error_message": "global_epoch必须是整数"})
    if not local_epoch.isdigit():
        return render(request, 'error.html', {"error_message": "local_epoch必须是整数"})
    client_num_list = ['5', '10']
    if client_num not in client_num_list:
        return render(request, 'error.html', {"error_message": "client_num不是限定值"})
    if chose_num > client_num:
        return HttpResponse("<script>alert('每轮选择节点数必须小于使用节点数');history.back(-1)</script>")


# 登出功能
def logout(request):
    request.session.delete()
    # 登出，则删除掉session中的某条数据
    # if 'is_login' in request.session:
    #     del request.session['is_login']
    # if 'is_remember' in request.session:
    #     del request.session['is_remember']
    return redirect("/login/")


# ----------------------view函数-------------------------
# 登陆页面

def login(request):
    # 记住登录信息，session直接登录
    if 'is_remember' in request.session and request.session['is_remember']:
        print(request.session['user_info'])
        useremail = request.session['user_info']['useremail']
        password_hash = request.session['user_info']['password_hash']
        try:
            userlist = models.UserLogin.objects.filter(uemail=useremail, password=password_hash)
            if userlist.exists():
                request.session['is_login'] = True

                return redirect('/home/')
        except:
            return render(request, 'error.html', {'error_message': '数据库查询错误'})

    if request.method == 'GET':
        return render(request, 'index.html')
    elif request.method == "POST":
        useremail = str(request.POST.get('Useremail'))
        password = str(request.POST.get('Password'))
        remember_sign = str(request.POST.get('remember'))

        # 正则匹配检查邮箱格式
        if not re.match(r'^[0-9a-za-z_]{0,19}@[0-9a-za-z]{1,13}\.[com,cn,net]{1,3}$', useremail):
            return HttpResponse("<script>alert('email 格式错误');history.back(-1)</script>")
            # return render(request, 'error.html')
        password_hash = MD5_hashCode(password)
        userlist = models.UserLogin.objects.filter(uemail=useremail, password=password_hash)
        # 判断用户是否存在
        if userlist.exists():
            username = userlist.values('username')[0]['username']
            request.session['user_info'] = {'username': username, 'useremail': useremail,
                                            'password_hash': password_hash}
            request.session['is_login'] = True
            # 设置session有效时间

            if remember_sign == 'on':
                request.session['is_remember'] = True
                request.session.set_expiry(60 * 60 * 24*1)

            else:
                request.session['is_remember'] = False
                request.session.set_expiry(0)
            return redirect('/home/')
        else:
            return HttpResponse("<script>alert('登录验证失败，请重新尝试');history.back(-1)</script>")


# 注册页面

def register(request):
    request.session.delete()
    if request.method == 'GET':
        return render(request, 'register.html')
    elif request.method == 'POST':
        re_list = [' ', 'and', '#', '--', 'union', 'select', '$', 'handler', '/**/']
        username = str(request.POST.get('username'))
        useremail = str(request.POST.get('email'))
        password = str(request.POST.get('password'))
        for aa in re_list:
            if aa in username:
                return HttpResponse("<script>alert('包含敏感字符，请从新输入');history.back(-1)</script>")
        # 邮箱格式检查
        if not re.match(r'^[0-9a-za-z_]{0,19}@[0-9a-za-z]{1,13}\.[com,cn,net]{1,3}$', useremail):
            return HttpResponse("<script>alert('email 格式错误');history.back(-1)</script>")
        email_list = models.UserLogin.objects.filter(uemail=useremail)
        username_list = models.UserLogin.objects.filter(username=username)
        if email_list.exists():
            return HttpResponse("<script>alert('email 已存在');history.back(-1)</script>")
        if username_list.exists():
            return HttpResponse("<script>alert('username 已存在');history.back(-1)</script>")
        password_hash = MD5_hashCode(password)
        userid = MD5_hashCode(useremail)
        try:
            models.UserLogin.objects.create(username=username, uemail=useremail, password=password_hash, userid=userid)
            return HttpResponse("<script>alert('注册成功，请返回登录');window.location.href ='/login/';</script>")
        except:
            return HttpResponse("<script>alert('用户创建错误');history.back(-1)</script>")


def user_logout(request):
    request.session.delete()
    return redirect("/login/")


# 主页
@login_check
def home(request):
    username, userid = get_session_userinfo(request)
    if request.method == "GET":

        sum_taskinfo_dic = []

        # 获取前端page数
        page = request.GET.get('home_page')
        if page:
            page = int(page)
        else:
            page = 1

        # 读取用户任务信息
        try:
            task_list = models.UserTask.objects.filter(userid=userid).values()
        except:
            return render(request, 'error.html', {"error_message": "usertask select error"})
        if task_list is not None:
            num = 1
            for i in task_list:
                taskid = i['taskid']
                try:
                    task_info = models.TaskInfo.objects.filter(taskid=taskid).values()[0]
                    task_info['page_id'] = num
                    num += 1
                    sum_taskinfo_dic.append(task_info)
                except:
                    print(taskid, "  home select wrong!")
            paginator = Paginator(sum_taskinfo_dic, 8)  # 每一页呈现出任务数量的个数
            page_num = paginator.num_pages
            if page > page_num:
                return render(request, 'error.html', {"error_message": "pagemnum error"})

            page_task_list = paginator.page(page)
            # 判断是否存在下一页
            if page_task_list.has_next():
                next_page = page + 1
            else:
                next_page = page
            # 是否存在上一页
            if page_task_list.has_previous():
                previous_page = page - 1
            else:
                previous_page = page
            return render(request, 'home.html', {
                'taskinfo_list': page_task_list,
                'curr_page': page,
                'page_num': page_num,
                'next_page': next_page,
                'previous_page': previous_page,
                'username': username,
                'userid': userid,
                'page': 'home',
                'search': 'no'
            })
        print(username)
        return render(request, 'home.html', {"username": username, "userid": userid, "page": "home", 'search': 'no'})
    # 当使用模糊查询的时候调用post方式响应
    if request.method == "POST":
        sum_taskinfo_dic = []
        serach_taskinfo_list = dict()
        select_taskid = request.POST.get('select_taskid')
        select_taskname = request.POST.get('select_taskname')
        select_algorithm = request.POST.get('select_algorithm')
        select_data_type = request.POST.get('select_data_type')
        select_deeplg_alg = request.POST.get('select_deeplg_alg')
        if select_taskid:
            serach_taskinfo_list['taskid'] = select_taskid
        if select_taskname:
            serach_taskinfo_list['taskname'] = select_taskname
        if select_algorithm:
            serach_taskinfo_list['algorithm'] = select_algorithm
        if select_data_type:
            serach_taskinfo_list['data_type'] = select_data_type
        if select_deeplg_alg:
            serach_taskinfo_list['deeplg_alg'] = select_deeplg_alg

        if serach_taskinfo_list is None:
            return HttpResponse("<script>alert('请只少输入一个查询值');history.back(-1)</script>")
        try:
            sum_taskinfo = models.TaskInfo.objects.filter(**serach_taskinfo_list).values()
        except:
            return render(request, 'error.html', {"error_message": "search taskinfo  sql error"})

        try:
            user_task_list = models.UserTask.objects.filter(userid=userid).values()
        except:
            return render(request, 'error.html', {"error_message": "search usertask sql error"})
        num = 1
        print(sum_taskinfo)
        print(user_task_list)
        user_taskid_list = []
        for item in user_task_list:
            user_taskid_list.append(item['taskid'])
        for i in sum_taskinfo:
            if i['taskid'] in user_taskid_list:
                i['page_id'] = num
                num += 1
                sum_taskinfo_dic.append(i)

        return render(request, 'home.html', {'userid': userid, "username": username,
                                             'search': 'yes', 'page': "home"
            , 'taskinfo_list': sum_taskinfo_dic
                                             })


# 删除任务响应
def task_del(request):
    if request.method == "POST":
        taskid = request.POST.get("taskid")
        filepath="./taskmod/"+taskid+".pt"
        if os.path.exists(filepath):
            os.remove(filepath)
        try:
            models.TaskInfo.objects.filter(taskid=taskid).delete()
            models.UserTask.objects.filter(taskid=taskid).delete()
            models.Taskdata.objects.filter(taskid=taskid).delete()

        except:
            return render(request, 'error.html', {'error_message': '数据库删除错误'})

        return redirect('/home/')


# 生成任务
@login_check
def create_task(request):
    username, userid = get_session_userinfo(request)

    if request.method == 'POST':

        taskname = request.POST.get("taskname")  # 任务名称
        global_epoch = request.POST.get("global_epoch")  # 全局迭代次数
        local_epoch = request.POST.get("local_epoch")  # 本地迭代次数
        algorithm = request.POST.get('algorithm')  # 联邦学习算法
        client_num = request.POST.get("client_num")  # 使用节点数
        chose_num = request.POST.get("chose_num")  # 每轮选择节点数
        task_info1 = {"taskname": taskname, "global_epoch": global_epoch, "local_epoch": local_epoch
            , "algorithm": algorithm, "client_num": client_num, "chose_num": chose_num}
        task_info_check(request, task_info1)
        task_list = models.TaskInfo.objects.filter(taskname=taskname)
        if task_list.exists():
            return HttpResponse("<script>alert('taskname 已存在');history.back(-1)</script>")
        return render(request, 'create_task2.html', {"task_info1": task_info1,
                                                     "username": username, "userid": userid
                                                     })
    elif request.method == 'GET':
        return render(request, 'create_task.html', {"username": username, "userid": userid, "page": "createtask"})
    return render(request, 'error.html', {"error_message": "method error"})


@login_check
def create_task2(request):
    username, userid = get_session_userinfo(request)
    if request.method == "POST":
        data_type_list = ["mnist", "kddcup99", "cifar"]
        # 前端传来的是str，转换成dic格式
        task_info1 = eval(request.POST.get("task_info1"))
        data_type = request.POST.get("data_type")
        deeplg_alg = request.POST.get("deeplg_alg")
        editordata = request.POST.get("editordata")

        # 再次对task_info1校验，防止中间篡改
        task_info_check(request, task_info1)
        if data_type not in data_type_list:
            return render(request, 'error.html', {'error_message': "wrong datatype"})

        # 对editordata进行xss攻击防护，后期补充

        taskname = task_info1['taskname']
        algorithm = task_info1['algorithm']
        global_epoch = task_info1['global_epoch']
        local_epoch = task_info1['local_epoch']
        client_num = task_info1['client_num']
        chose_num = task_info1['chose_num']

        task_id = MD5_hashCode(taskname)

        # 针对taskname进行重复性检验
        task_list = models.TaskInfo.objects.filter(taskname=taskname)
        if task_list.exists():
            return HttpResponse("<script>alert('taskname 已存在');history.back(-1)</script>")
        # 建立user_id 与 task_id之间的联系
        try:
            models.UserTask.objects.create(userid=userid, taskid=task_id)
        except:
            return render(request, 'error.html', {"error_message": "usertask input error"})
        # taskinfo 写入数据库
        try:
            models.TaskInfo.objects.create(taskname=taskname, taskid=task_id, description=editordata,
                                           global_iteration=global_epoch
                                           , local_iteration=local_epoch, algorithm=algorithm, status=0,
                                           client_num=client_num, chose_num=chose_num,
                                           data_type=data_type, deeplg_alg=deeplg_alg)
            return render(request, 'create_task3.html', {"username": username, "taskid": task_id})
        except:
            return render(request, 'error.html', {"error_message": "taskinfo input error"})

    return render(request, 'error.html', {"error_message": "method error"})


@login_check
def taskdata(request):
    username, userid = get_session_userinfo(request)
    if request.method == "GET":
        acc = [0]
        loss = [1]
        attack_rate = []
        taskid = request.GET.get('taskid')
        taskdata = models.Taskdata.objects.filter(taskid=taskid)
        taskinfo = models.TaskInfo.objects.filter(taskid=taskid)
        if taskdata.exists() and taskinfo.exists():
            taskinfo=taskinfo.values()[0]
            taskdata=taskdata.values()[0]
            data = eval(taskdata['data']).values()
            epochs =['0']+ list(eval(taskdata['data']).keys())
            for item in data:
                acc.append(item[0])
                loss.append(item[1])
                attack_rate.append(item[2])
            if taskinfo['algorithm']=='VFed-avg':
                # 计算vote
                svote=list(eval(taskdata['svote']).values())
                votedata = ['client_' + str(i+1) for i in range(len(svote))]
                print(svote)
                cvote=list(eval(taskdata['cvote']).values())
                # 构造json
                json_server="["
                json_client = "["
                for i in range(len(svote)):
                    json_server+="{value:"+str(svote[i])+",name:'"+votedata[i]+"'},"
                    json_client+="{value:"+str(cvote[i])+",name:'"+votedata[i]+"'},"
                json_server+="]"
                json_client+="]"

                return render(request,'taskpanel.html',{'processing':100,'json_client':json_client,'json_server':json_server,'votedata':votedata,'taskinfo':taskinfo,'acc':acc,'loss':loss,'epochs':epochs,"page":"taskqu","username": username, "userid": userid})

            elif taskinfo['algorithm']=='Fed-avg':
                return render(request,'taskpanel.html',{'processing':100,'taskinfo':taskinfo,'acc':acc,'loss':loss,'epochs':epochs,"page":"taskqu","username": username, "userid": userid})
        else:
            taskinfo = taskinfo.values()[0]
            epochs=['0']
            acc=[0]
            loss=[1]
            if taskinfo['status'] == 0:
                processing=10
            elif taskinfo['status'] == 1:
                processing=50
            if taskinfo['algorithm'] == 'VFed-avg':
                clients=taskinfo['client_num']
                json_server = "["
                json_client = "["
                votedata = ['client_' + str(i + 1) for i in range(clients)]
                for i in range(clients):
                    json_server+="{value:0 ,name:'"+votedata[i]+"'},"
                    json_client+="{value:0 ,name:'"+votedata[i]+"'},"
                json_server+="]"
                json_client+="]"
                return render(request,'taskpanel.html',{'processing':processing,'json_client':json_client,'json_server':json_server,'votedata':votedata,'taskinfo':taskinfo,'acc':acc,'loss':loss,'epochs':epochs,"page":"taskqu","username": username, "userid": userid})
            elif taskinfo['algorithm']=='Fed-avg':
                return render(request,'taskpanel.html',{'processing':processing,'taskinfo':taskinfo,'acc':acc,'loss':loss,'epochs':epochs,"page":"taskqu","username": username, "userid": userid})


# 下载模型文件
@login_check
def getfile(request ,taskid):
    # 服务器上存放文件的路径
    file_path = r"./taskmod/"+taskid+".pt"
    file_name=taskid+".pt"
    print(file_path)
    with open(file_path, 'rb') as f:
        try:
            response = HttpResponse(f)
            response['content_type'] = "application/octet-stream"
            response['Content-Disposition'] = 'attachment; filename=' + file_name
            return response
        except Exception:
            raise Http404


# 点击任务面板详情
@login_check
def taskqu(request):
    username, userid = get_session_userinfo(request)
    if request.method == 'POST':
        taskid=request.POST.get('taskid')
        return HttpResponseRedirect("/taskdata/?taskid="+taskid)
    elif request.method == "GET":
        return render(request,'taskinputid.html',{"username": username, "userid": userid,'page':"taskqu"})



# ########################### celery异步数据处理 ####################
from .task import *
def taskstart(request):
    if request.method == "POST":
        taskid = request.POST.get('startid')
        # 调整任务状态为执行状态
        models.TaskInfo.objects.filter(taskid=taskid).update(status='1')
        b = flrun.delay(taskid)
        return HttpResponse("<script>alert('"+b.id+"任务开始执行');window.location.replace(document.referrer);</script>")
    # return render(request,"test.html")

# ##########################邮箱验证码发送##########################
def findpwd(request):
    if request.method=="GET":
        return render(request,'getpassword.html')
    elif request.method=="POST":
        varycode="******"
        if 'verify' in request.session:
            varycode=request.POST.get("vertify")
        email=request.POST.get("email")
        newpassword=request.POST.get("newpassword")
        password_hash = MD5_hashCode(newpassword)
        code=request.session['verify']
        btime=request.session['veryfytime']
        btime=datetime.datetime.strptime(btime, "%Y-%m-%d %H:%M:%S")
        ntime=datetime.datetime.now()
        second=(ntime-btime).total_seconds()
        if second >60:
            if 'verify' in request.session:
                del request.session['verify']
            return HttpResponse("<script>alert('验证码超时，请重新获取');history.back(-1)</script>")
        if varycode==code:
            try:
                models.UserLogin.objects.filter(uemail=email).update(password=password_hash)
                if 'verify' in request.session:
                    del request.session['verify']
                return HttpResponse("<script>alert('密码修改成功');window.location.href ='/login/';</script>")
            except:
                return HttpResponse("<script>alert('密码修改成功');history.back(-1)</script>")
        else:
            return HttpResponse("<script>alert('验证码错误，请返回输入正确验证码');history.back(-1)</script>")
#发送邮件
def sendMessage(email):#发送邮件并返回验证码
    # 生成验证码
    import random
    str1 = '0123456789'
    rand_str = ''
    for i in range(0, 6):
        rand_str += str1[random.randrange(0, len(str1))]
    # 发送邮件：
    # send_mail的参数分别是  邮件标题，邮件内容，发件箱(settings.py中设置过的那个)，收件箱列表(可以发送给多个人),失败静默(若发送失败，报错提示我们)
    message = "您的验证码是" + rand_str + "，10分钟内有效，请尽快填写"
    emailBox = []
    emailBox.append(email)
    send_mail("找回密码", message, "yangty0055@163.com", emailBox, fail_silently=False)
    return rand_str

# 验证该用户是否已存在 created = 1 存在
def existUser(email):
    created = 1
    try:
        models.UserLogin.objects.get(uemail=email)
    except:
        created = 0
    return created


def reg(request):
    if request.POST.get('type') == 'sendMessage':

        print(request.POST.get('email'))
        email = request.POST.get('email')
        response = {"state": False, "errmsg": ""}
        if not existUser(email):
            response['errmsg'] = '该邮箱不存在，请注册'
        else:

            rand_str = sendMessage(email)#发送邮件
            request.session['verify'] = rand_str # 验证码存入session，用于做注册验证
            request.session['veryfytime']=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response['state'] = True
            response['errmsg']='验证码已发送，有效期10分钟'
            # except:
            #     response['errmsg'] = '验证码发送失败，请检查邮箱地址'
            return HttpResponse(json.dumps(response))
