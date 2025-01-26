# -*- coding: UTF-8 -*-

import base64
import os
import time
import pymysql
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect,session
from yolov5 import YOLOv5, video_detect_generator
from flask_session import Session
app = Flask(__name__)
# 字符串随便起
app.secret_key = "affedasafafqwe"
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
yolo = YOLOv5()


@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# 允许使用的图片文件格式
ALLOWED_IMAGE_EXTENSIONS = {'bmp', 'dng', 'jpeg',
                            'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'}
# 允许使用的视频文件格式
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi'}


def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    img = request.files.get('file')
    if img is None or img.filename == '':
        return jsonify({'code': -1, 'msg': '图片不能为空'})
    if allowed_image_file(img.filename) is False:
        return jsonify({'code': -1, 'msg': '图片格式不支持'})
    suffix = img.filename.split('.')[-1]
    img = img.read()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    xyxys, confs, cls_ids, class_counts, img_copy = yolo.infer_and_draw(img)
    file_name = str(int(time.time())) + '.' + suffix
    file_path = './static/download/' + file_name
    cv2.imwrite(file_path, img)

    img = cv2.imencode(f'.{suffix}', img)[1].tobytes()
    img_copy = cv2.imencode(f'.{suffix}', img_copy)[1].tobytes()

    return jsonify({
        'code': 0,
        'msg': 'success',
        'predict': base64.b64encode(img).decode('utf-8'),
        'source': base64.b64encode(img_copy).decode('utf-8'),
        'fileName': file_name,
        'suffix': suffix,
        'classCounts': class_counts,
        'xyxys': xyxys,
        'confs': confs,
        'classIds': cls_ids,
    })


@app.route('/download', methods=['GET'])
def download():
    file_name = request.args.get('fileName')
    if allowed_image_file(file_name) is True:
        msg = '图片'
    elif allowed_video_file(file_name) is True:
        msg = '视频'
    elif file_name.isnumeric():
        msg = '摄像头'
        file_name = file_name + '.mp4'
    else:
        return jsonify({'code': -1, 'msg': '文件格式不支持'})

    file_path = os.path.join('./static/download/', file_name)
    if not os.path.isfile(file_path):
        return jsonify({'code': -1, 'msg': msg + '不存在'})
    return jsonify({
        'code': 0,
        'msg': msg + '下载成功',
        'url': '/static/download/' + file_name
    })


@app.route('/uploadVideo', methods=['POST', 'GET'])
def upload_video():
    if request.method == 'GET':
        return render_template('video.html')
    video = request.files.get('file')
    if video is None or video.filename == '':
        return jsonify({'code': -1, 'msg': '视频不能为空'})
    if allowed_video_file(video.filename) is False:
        return jsonify({'code': -1, 'msg': '视频格式不支持'})
    suffix = video.filename.split('.')[-1]
    video = video.read()
    video_name = str(int(time.time())) + '.' + suffix
    with open('./static/upload/' + video_name, 'wb') as f:
        f.write(video)
    return jsonify({
        'code': 0,
        'msg': '视频上传成功',
        'videoName': video_name,
    })


video_generator_map = {}


@app.route('/videoPredict', methods=['GET'])
def video_predict():
    global video_generator_map
    video_name = request.args.get('videoName')# type: str

    if video_name.isnumeric():
        video_path = int(video_name)
        video_name = video_name + '.mp4'
        if video_name not in video_generator_map:
            video = cv2.VideoCapture(video_path)
            ret, frame = video.read()
            video.release()
            if ret is False:
                return jsonify({'code': -1, 'msg': '摄像头不存在'})
    else:
        video_path = './static/upload/' + video_name
        if not os.path.isfile(video_path):
            return jsonify({'code': -1, 'msg': '视频不存在'})

    is_stop = request.args.get('isStop')
    if is_stop == 'true':
        video_generator_map[video_name].close()
        del video_generator_map[video_name]
        return jsonify({'code': 1, 'msg': '视频检测停止'})

    if video_name not in video_generator_map:
        video_generator_map[video_name] = video_detect_generator(
            yolo, video_path, video_name)
    video_generator = video_generator_map[video_name]
    try:
        result = next(video_generator)
    except StopIteration:
        del video_generator_map[video_name]
        return jsonify({'code': 1, 'msg': '视频检测完毕'})

    frame = result['frame']
    frame_copy = result['frame_copy']
    frame = cv2.imencode('.jpg', frame)[1].tobytes()
    frame_copy = cv2.imencode('.jpg', frame_copy)[1].tobytes()
    # 返回图片
    return jsonify({
        'code': 0,
        'msg': '视频检测成功',
        'predict': base64.b64encode(frame).decode('utf-8'),
        'source': base64.b64encode(frame_copy).decode('utf-8'),
        'filePath': video_path,
        'suffix': 'jpg',
        'classCounts': result['class_counts'],
        'xyxys': result['xyxys'],
        'confs': result['confs'],
        'classIds': result['cls_ids'],
    })


@app.route('/camera', methods=['GET'])
def camera():
    return render_template('camera.html')


@app.route('/login',methods=['GET','POST'])
def iogin():
    if request.method == "POST":
        name = request.form.get('username')
        password = request.form.get('password')
        conn = pymysql.connect(host='127.0.0.1', user='root', passwd='root', port=3306, db='yz1', charset='utf8')

        # 创建游标对象
        cursor = conn.cursor()
        sql = 'select id from `login` where name = "{0}" and password = "{1}"  '.format(name, password)
        cursor.execute(sql)
        res=cursor.fetchall()
        if res == ():
            data = {
                'info': '该用户未注册，请注册后在登录'
            }
            return data
        else:
            data = {
                'info': "登录成功",
                'userid': res[0][0]
            }
        print(data)
        print(name)
        print(password)
        if data['info'] == '登录成功':
            session['userid'] = data['userid']
            return redirect('index')
        else:
            return redirect('/')

@app.route('/',methods=['GET','POST'])
def ind():
    # if request.session.get('is_login', None):

    #     return redirect("/")


        # if login_form.is_valid():
        #     userID = login_form.cleaned_data['userID']
        #     password = login_form.cleaned_data['password']
        #     try:
        #         user = models.Account.objects.get(userID=userID)
        #         if user.password == password:  # 哈希值和数据库内的值进行比对
        #             request.session['is_login'] = True
        #             request.session['user_id'] = user.userID
        #             request.session['identify'] = user.identity
        #             return redirect('/')
        #         else:
        #             message = "密码不正确！"
        #     except:
        #         message = "该账户不存在！"

    # return render_template('/')
    return render_template( 'login.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
