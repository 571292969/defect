# -*- coding: utf-8 -*-

import tkinter as tk
from PIL import Image,ImageTk
import easygui
import colorsys
import os
from os import listdir
from os.path import join
from timeit import default_timer as timer
import time
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pymysql
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

import time
import tkinter.ttk
#path = '2012_test.txt'
path = 'model_data/test/'  # 待检测图片的位置
count = 0
# 创建创建一个存储检测结果的dir
result_path = 'model_data/result'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# result如果之前存放的有文件，全部清除
for i in os.listdir(result_path):
    path_file = os.path.join(result_path, i)
    if os.path.isfile(path_file):
        os.remove(path_file)

# 创建一个记录检测结果的文件
txt_path = result_path + '/result.txt'
file = open(txt_path, 'w')


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/trained_weights_final.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (512, 512),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, portion):
        start = timer()  # 开始计时
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)  # 打印图片的尺寸
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 提示用于找到几个bbox

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(2e-2 * image.size[1] + 0.2).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        # 保存框检测出的框的个数
        file.write('find  ' + str(len(out_boxes)) + ' target(s) \n')
        global count
        count += len(out_boxes)
        db = pymysql.connect(host = '10.20.8.104',user = 'cjx',password = 'cjx571292969',database = 'defect',charset="utf8", autocommit=True)
        cursor = db.cursor()    
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            
            sql = "SELECT * FROM defect.defect_start;"
        
            try:
                cursor.execute(sql)
                results = cursor.fetchall()
            except:
                print("Error: unable to fetch data")
            
            flot = 0
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            
            for row in results:
                defect_id = row[0]
                name = row[1]
                x = row[2]
                y = row[3]
                defect_start = row[4]
                flot_0 = 0 
                #print(defect_id,name,x,y,defect_start)
                if (((right + left)/2 - x)**2 + ((bottom + top)/2 - y)**2 <= 900) and (name ==predicted_class):
                    label = '{} {:.2f}'.format(predicted_class + str(defect_id), score)   #标记框的名称
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)
                    flot = 1
                    break
                else:
                    for j, b in reversed(list(enumerate(out_classes))):
                        predicted_class_0 = self.class_names[b]
                        box_0 = out_boxes[j]
                        score_0 = out_scores[j]
                    
                        top_0, left_0, bottom_0, right_0 = box_0
                        top_0 = max(0, np.floor(top_0 + 0.5).astype('int32'))
                        left_0 = max(0, np.floor(left_0 + 0.5).astype('int32'))
                        bottom_0 = min(image.size[1], np.floor(bottom_0 + 0.5).astype('int32'))
                        right_0 = min(image.size[0], np.floor(right_0 + 0.5).astype('int32'))
                        
                        if(((right_0 + left_0)/2 - x)**2 + ((bottom_0 + top_0)/2 - y)**2 <= 900) and (name ==predicted_class_0):
                            flot_0 = 1
                            break
                    if flot_0 == 0:
                        end_id = defect_id
                        end_name = name
                        end_x = x
                        end_y = y
                        defect_end = portion[1].strip(".jpg")
                        
                        sql_end_insert = "INSERT INTO defect_end(defect_id, name, X, Y, defect_end) VALUES(%s,'%s',%s,%s,'%s')" % (end_id,end_name,end_x,end_y,defect_end)
                        cursor.execute(sql_end_insert)
                        #print("insert")
                        
                        sql_start_delete = "DELETE FROM defect_start where defect_id = %s" % (defect_id)
                        cursor.execute(sql_start_delete)
                        #print('delete')
                        
                        sql_whole_end = "UPDATE defect_whole SET defect_end = '%s' WHERE defect_id = '%s'" % (defect_end, end_id)
                        cursor.execute(sql_whole_end)
                    
                
            if flot == 0:
                if len(results) == 0:
                    ID = 1
                else:
                    ID = defect_id + 1
                Name = predicted_class
                X = (right + left) / 2
                Y = (bottom + top) / 2
                Defect_start = portion[1].strip(".jpg")
                #print(ID,Name,X,Y,Defect_start)
                sql_insert = "INSERT INTO defect_start(defect_id, name, X, Y, defect_start) VALUES(%s,'%s',%s,%s,'%s')" % (ID,Name,X,Y,Defect_start)
                cursor.execute(sql_insert)
                sql_whole_start = "INSERT INTO defect_whole(defect_id, defect_name, defect_x, defect_y, defect_start) VALUES(%s,'%s',%s,%s,'%s')" % (ID,Name,X,Y,Defect_start)
                cursor.execute(sql_whole_start)
                label = '{} {:.2f}'.format(predicted_class + str(ID), score)   #标记框的名称
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

            file.write(
                predicted_class + '  score: ' + str(score) + ' \nlocation: top: ' + str(top) + '、 bottom: ' + str(
                    bottom) + '、 left: ' + str(left) + '、 right: ' + str(right) + '\n####  center location:' + '(' + str((right + left)/2) + ',' + str((bottom + top)/2) + ')  ####'+'\n\n')

            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print('time consume:%.3f s ' % (end - start))
        file.write('\n-----------------------------\n')
        return image

    def close_session(self):
        self.sess.close()


# 界面操作
count_image = 1
path_image = "model_data/test/"

class baseWindow():
    def __init__(self,master):
        self.root = master
        self.root.configure()
        self.root.title("缺陷检测")
        self.root.geometry("600x600")
        btbase = tk.Button(self.root, text="上传图像", command=self.upload, height=3, width=20,font=("宋体", 16))
        btbase.grid(row=1, column=0)
        btbase.place(relx=0.5, rely=0.5, anchor="center")

    def is_image_file(self,filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def upload(self,event = None):
        imagePath = easygui.diropenbox("选择文件")
        print(imagePath)

        if(imagePath != None):
            for i in os.listdir(path_image):
                path_file = os.path.join(path_image, i)
                if os.path.isfile(path_file):
                    os.remove(path_file)

            files = os.listdir(imagePath)

            self.progressbarOne = tkinter.ttk.Progressbar(self.root)
            self.progressbarOne.place(relx=0.5, rely=0.8, anchor="center")

            self.progressbarOne['max'] = len(files)
            self.progressbarOne['value'] = 0

            image_filenames = [join(imagePath, x) for x in listdir(imagePath) if self.is_image_file(x)]
            self.a = 1
            for image_filename in image_filenames:
                im = Image.open(image_filename)
                image_name = str(self.a).zfill(6) + '.jpg'
                save = os.path.abspath('.') + "\\" + "model_data\\test\\" + image_name
                print(save)
                im.save(save)
                self.a += 1
                self.progressbarOne['value'] = self.a + 1
                self.root.update()

            self.root.destroy()
            defectWindow(self)


class defectWindow():

    def __init__(self,master):
        self.master = master
        self.defect_win = tk.Tk()
        self.defect_win.title("缺陷识别")
        self.defect_win.geometry("1600x800")
        pil_image = Image.open(path_image + str(count_image).zfill(6) + '.jpg')
        w_box = 512
        h_box = 512
        w, h = pil_image.size

        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)

        photo = ImageTk.PhotoImage(pil_image_resized)
        self.imageLabel = tk.Label(self.defect_win, image=photo, width=w_box, height=h_box)
        self.imageLabel.place(x=100, y=100)

        files = os.listdir(path_image)
        self.imagenum = len(files)
        # 图片切换
        btnup = tk.Button(self.defect_win, text="<<", command=self.up, height=1, width=4)
        btndown = tk.Button(self.defect_win, text=">>", command=self.down, height=1, width=4)
        btnup.grid(row=1, column=0)
        btndown.grid(row=1, column=1)
        btnup.place(x=45, y=340)
        btndown.place(x=632, y=340)

        # 导航窗格
        self.image_now = tk.Label(self.defect_win, text="共 ["+str(self.imagenum)+"] 张图片；当前图片为："+str(count_image).zfill(6),font=("宋体",12))
        self.image_now.place(x = 100,rely=0.08)
        # 菜单
        mainmenu = tk.Menu(self.defect_win)
        menuOption = tk.Menu(mainmenu, tearoff=False)
        mainmenu.add_cascade(label="选项", menu=menuOption)
        menuOption.add_command(label="上传", command=self.upload2, accelerator="Ctrl + N")
        menuOption.add_command(label="导出结果", command=self.export, accelerator="Ctrl + E")

        menuOption2 = tk.Menu(mainmenu,tearoff=False)
        mainmenu.add_cascade(label="帮助", menu=menuOption2)

        menuOption.add_separator()
        menuOption.add_command(label="退出", command=self.defect_win.destroy)
        self.defect_win.configure(menu=mainmenu)
        self.defect_win.bind("<Control-n>", self.upload2)
        self.defect_win.bind("<Control-N>", self.upload2)
        self.defect_win.bind("<Control-E>", self.export)
        self.defect_win.bind("<Control-e>", self.export)

        # 识别按钮
        btdistinguish = tk.Button(self.defect_win, text="识别", command=self.distinguish,font=("宋体",14))
        btdistinguish.grid(row=1, column=2)
        btdistinguish.place(x=512, y=650,height=30,width=100)
        # 上传按钮
        btupload = tk.Button(self.defect_win,text="上传",command=self.upload2,font=("宋体",14))
        btupload.grid(row=1,column=3)
        btupload.place(x=412,y=650,height=30,width=100)
        # 模型选择
        comvalue = tk.StringVar()
        self.btmodel = tk.ttk.Combobox(self.defect_win,font=("宋体",14),state="readonly",textvariable=comvalue)
        self.btmodel["values"] = ("点阵结构1","点阵结构2","点阵结构3","点阵结构4")
        self.btmodel.current(0)
        self.btmodel.bind("<<ComboboxSelected>>",self.model)
        self.btmodel.place(x = 200,y = 650,height=30,width=150)

        model_label = tk.Label(self.defect_win,text="模型：",font=("宋体",14))
        model_label.place(x = 100, y = 650,height=30,width=100)

        self.imageLabel.configure(image=photo)
        self.imageLabel.image = photo

    def is_image_file2(self,filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def upload2(self,event=None):
        imagePath = easygui.diropenbox("选择文件")
        print(imagePath)
        global path_image,count_image
        count_image = 1
        path_image = "model_data/test/"
        if(imagePath != None):
            for i in os.listdir(path_image):
                path_file = os.path.join(path_image, i)
                if os.path.isfile(path_file):
                    os.remove(path_file)

            files = os.listdir(imagePath)

            self.progressbarOne = tkinter.ttk.Progressbar(self.defect_win)
            self.progressbarOne.place(relx=0.19, y = 350)

            self.progressbarOne['max'] = len(files)
            self.progressbarOne['value'] = 0

            image_filenames = [join(imagePath, x) for x in listdir(imagePath) if self.is_image_file2(x)]
            self.b = 1
            for image_filename in image_filenames:
                im = Image.open(image_filename)
                image_name = str(self.b).zfill(6) + '.jpg'
                save = os.path.abspath('.') + "\\" + "model_data\\test\\" + image_name
                print(save)
                im.save(save)
                self.b += 1
                self.progressbarOne['value'] = self.b + 1
                self.defect_win.update()

            self.defect_win.destroy()
            defectWindow(self)

    def resize(self,w, h, w_box, h_box, pil_image):
        f1 = 1.0 * w_box / w
        f2 = 1.0 * h_box / h
        factor = min([f1, f2])

        width = int(w * factor)
        height = int(h * factor)

        return pil_image.resize((width, height), Image.ANTIALIAS)

    def distinguish(self):
        global path_image
        path_image = "model_data/test/"
        files = os.listdir(path_image)

        self.progressbarOne = tkinter.ttk.Progressbar(self.defect_win, length=1600)
        self.progressbarOne.place(x=0, rely=0.95)

        self.progressbarOne['max'] = len(files)
        self.progressbarOne['value'] = 0

        self.defect_progress = 0
        self.defect_win.update()
        t1 = time.time()
        yolo = YOLO()

        # 清空数据库原本数据
        db = pymysql.connect(host='10.20.8.104', user='cjx', password='cjx571292969', database='defect', charset="utf8",
                             autocommit=True)
        cursor = db.cursor()

        sql_delete_start_all = "DELETE From defect_start where defect_id > 0"
        cursor.execute(sql_delete_start_all)
        sql_delete_end_all = "DELETE From defect_end where defect_id > 0"
        cursor.execute(sql_delete_end_all)
        sql_delete_whole_all = "DELETE From defect_whole where defect_id > 0"
        cursor.execute(sql_delete_whole_all)


        # 开始识别循环
        for filename in sorted(os.listdir(path)):
            image_path = path + '/' + filename
            portion = os.path.split(image_path)
            file.write(portion[1] + ' detect_result：\n')
            image = Image.open(image_path)
            r_image = yolo.detect_image(image, portion)
            file.write('\n')
            # r_image.show() #显示检测结果
            image_save_path = 'model_data/result/result_' + portion[1]
            print('detect result save to....:' + image_save_path)
            r_image.save(image_save_path)
            self.progressbarOne['value'] = self.defect_progress + 1
            self.defect_win.update()
            self.defect_progress += 1

        #  结果输出
        time_sum = time.time() - t1
        file.write('time sum: ' + str(time_sum) + 's')
        print('time sum:', time_sum)
        file.write('target sum: ' + str(count))
        print('target sum: ', count)
        #file.close()
        #yolo.close_session()

        path_image = "model_data/result/result_"
        pil_image = Image.open(path_image + str(count_image).zfill(6) + '.jpg')
        w_box = 512
        h_box = 512
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)
        photo = ImageTk.PhotoImage(pil_image_resized)
        self.imageLabel.configure(image=photo)
        self.imageLabel.image = photo
        self.progressbarOne.destroy()
        self.defect_win.update()

        sql_select_whole = "SELECT * FROM defect.defect_whole"
        cursor.execute(sql_select_whole)
        results = cursor.fetchall()
        self.whole_count = 0
        tree = tk.ttk.Treeview(self.defect_win)
        tree["show"] = 'headings'
        tree["columns"] = ("缺陷ID","缺陷名称","截面横坐标","截面纵坐标","缺陷开始图像","缺陷结束图像")
        tree.column("缺陷ID",width=100)
        tree.column("缺陷名称",width=100)
        tree.column("截面横坐标",width=100)
        tree.column("截面纵坐标",width=100)
        tree.column("缺陷开始图像",width=100)
        tree.column("缺陷结束图像",width=100)
        tree.heading("缺陷ID",text="缺陷ID")
        tree.heading("缺陷名称", text="缺陷名称")
        tree.heading("截面横坐标", text="截面横坐标")
        tree.heading("截面纵坐标", text="截面纵坐标")
        tree.heading("缺陷开始图像", text="缺陷开始图像")
        tree.heading("缺陷结束图像", text="缺陷结束图像")
        for row in results:
            defect_id = row[0]
            defect_name = row[1]
            defect_x = row[2]
            defect_y = row[3]
            defect_start = row[4]
            defect_end = row[5]
            tree.insert("",self.whole_count,values=(defect_id,defect_name,defect_x,defect_y,defect_start,defect_end))
            self.whole_count += 1
        tree.place(x=900,y=100,height=512)

        sql_unique_defect = "SELECT DISTINCT defect_name FROM defect_whole"
        cursor.execute(sql_unique_defect)
        results_type = cursor.fetchall()
        self.type = ""
        for row in results_type:
            defect_type = row[0]
            self.type += "["+str(defect_type)+"]"

        self.time_label = tk.Label(self.defect_win,text="总用时: [ "+str(time_sum)[0:5]+" ]    缺陷总数： [ " + str(len(results)) + " ]    缺陷种类： [ 共 " + str(len(results_type))+" 种 ]",font=("宋体",12))
        self.time_label.place(x = 900,rely=0.08)

        sql_type_count = "SELECT defect_name, count(*) as count from defect_whole group by defect_name"
        cursor.execute(sql_type_count)
        results_type_count = cursor.fetchall()
        self.type_count = 0
        tree_type = tk.ttk.Treeview(self.defect_win)
        tree_type["show"] = 'headings'
        tree_type["columns"] = ("缺陷类型","数量")
        tree_type.column("缺陷类型",width=100)
        tree_type.column("数量",width=100)
        tree_type.heading("缺陷类型",text="缺陷类型")
        tree_type.heading("数量",text="数量")
        for row in results_type_count:
            type_name = row[0]
            type_count = row[1]
            tree_type.insert("",self.type_count,values=(type_name,type_count))
            self.type_count += 1
        tree_type.place(x = 1300,y = 650, height=100)

        db.close()

    def up(self):
        global count_image
        count_image -= 1
        if count_image < 1:
            count_image = self.imagenum

        pil_image = Image.open(path_image + str(count_image).zfill(6) + '.jpg')
        # print(path_image + str(count_image).zfill(6) + '.jpg')
        w_box = 512
        h_box = 512
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)
        photo = ImageTk.PhotoImage(pil_image_resized)
        self.imageLabel.configure(image=photo)
        self.imageLabel.image = photo
        self.image_now.configure(text="共 ["+str(self.imagenum)+"] 张图片；当前图片为："+str(count_image).zfill(6))

    def down(self):
        global count_image
        count_image += 1
        if count_image > self.imagenum:
            count_image = 1

        pil_image = Image.open(path_image + str(count_image).zfill(6) + '.jpg')
        # print(path_image + str(count_image).zfill(6) + '.jpg')
        w_box = 512
        h_box = 512
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)
        photo = ImageTk.PhotoImage(pil_image_resized)
        self.imageLabel.configure(image=photo)
        self.imageLabel.image = photo
        self.image_now.configure(text="共 [" + str(self.imagenum) + "] 张图片；当前图片为：" + str(count_image).zfill(6))

    def model(self,*args):
        print(self.btmodel.get())

    def export(self,event=None):
        s = '导出'

    def help(self):
        s = '帮助'
if __name__ == '__main__':

    root = tk.Tk()
    baseWindow(root)
    root.mainloop()

