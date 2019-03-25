# this file is to detect which number effects to the images

import sys
import glob
import random
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QSlider
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt

from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import chainer
from chainer import serializers
from chainer import cuda
from keras.models import load_model
import network
import utils

depth = 6
gen_path = '../eiki_generator/results/gen'
vec2rand_model_path = './model/vec2rand.model'
gpu = -1
csv_len = 13


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 image - pythonspot.com'
        self.left = 300
        self.top = 300
        self.width = 900
        self.height = 600
        self.gen = network.Generator(depth=depth)
        serializers.load_npz(gen_path, self.gen)
        # file
        self.csv_file = [[random.random() for j in range(csv_len)]]
        self.vec2rand_model = load_model(vec2rand_model_path)
        # gpu using
        if gpu >= 0:
            cuda.get_device_from_id(0).use()
            self.gen.to_gpu()
        self.xp = self.gen.xp
        self.initUI()
        #generating image.
        z = self.xp.random.randn(1, 512, 1, 1).astype('f')
        x = self.gen(z, alpha=1.0)
        x = chainer.cuda.to_cpu(x.data)
        img = x[0].copy()
        utils.save_image(img, 'temp.jpg')
        _img = Image.open('temp.jpg')
        self.img = np.asarray(_img)
        self.initFigure()
        self.show()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # UI 調整
        lbl1 = self.makelabel('右腕上げ', (50,50))
        self.ude_migi_age = self.cbutton('右腕上げ',(50,100),'r_arm_up')
        lbl1 = self.makelabel('左腕上げ', (150,50))
        self.ude_hidari_age = self.cbutton('左腕上げ',(150,100),'l_arm_up')
        lbl1 = self.makelabel('右腕中', (50,150))
        self.ude_migi_tyuu = self.cbutton('右腕中', (50, 200), 'r_arm_tyuu')
        lbl1 = self.makelabel('左腕中', (150,150))
        self.ude_hidari_tyuu = self.cbutton('左腕中',(150,200), 'l_arm_tyuu')
        lbl1 = self.makelabel('右腕下', (50,250))
        self.ude_migi_sita = self.cbutton('右腕下', (50, 300), 'r_arm_sita')
        lbl1 = self.makelabel('左腕下', (150,250))
        self.ude_hidari_sita = self.cbutton('左腕下',(150,300), 'l_arm_sita')
        lbl1 = self.makelabel('顔右', (250,50))
        self.kao_migi_muki = self.cbutton('顔右',(250,100),'r_face')
        lbl1 = self.makelabel('顔左', (250,150))
        self.kao_hidari_muki = self.cbutton('顔左',(250,200),'l_face')
        lbl1 = self.makelabel('顔上', (250,250))
        self.kao_ue_muki = self.cbutton('顔上',(250,300),'u_face')
        lbl1 = self.makelabel('顔下', (250,350))
        self.kao_shita_muki = self.cbutton('顔下',(250, 400),'d_face')
        lbl1 = self.makelabel('顔正面', (250,450))
        self.kao_shoumen_muki = self.cbutton('顔正面', (250, 500), 'c_face')
        lbl1 = self.makelabel('崩壊補正', (350,50))
        self.houkai = self.cbutton('崩壊補正', (350, 100), 'c_face')
        lbl1 = self.makelabel('まばたき', (350,150))
        self.mabataki = self.cbutton('まばたき',(350,200),'mabataki')
        self.tugihe = self.cbutton('ランダム描写', (350, 250), 'random')
        self.tugihe = self.cbutton('描写', (350, 300), 'next')
        close = self.cbutton('終了', (350, 350), 'fin')
        torikesi = self.cbutton('初期化',(350,400),'torikesi')
        save = self.cbutton('保存', (350,450), 'save')


        # Create widget
        self.FigureWidget = QWidget(self)
        self.FigureLayout = QVBoxLayout(self.FigureWidget)
        self.FigureLayout.setContentsMargins(0,0,0,0)
        self.FigureWidget.setGeometry(450,150,450,214)

    def makelabel(self, title, geometry):
        lbl = QLabel(title, self)
        lbl.move(geometry[0], geometry[1])

    def initFigure(self):
        # Figureを作成
        self.Figure = plt.figure()
        # FigureをFigureCanvasに追加
        self.FigureCanvas = FigureCanvas(self.Figure)
        # LayoutにFigureCanvasを追加
        self.FigureLayout.addWidget(self.FigureCanvas)

        self.axis = self.Figure.add_subplot(1,1,1)
        self.axis_image = self.axis.imshow(self.img, cmap='gray')
        plt.axis('off')

    # Figureを更新
    def update_Figure(self, img):
        self.axis_image.set_data(img)
        self.FigureCanvas.draw()


    def cbutton(self, title, geometry, type):
        button = None
        if type == 'r_arm_up':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'l_arm_up':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'r_arm_tyuu':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'l_arm_tyuu':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'r_arm_sita':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'l_arm_sita':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'r_face':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'l_face':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'u_face':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'd_face':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'c_face':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'next':
            button = QPushButton(title, self)
            button.move(geometry[0], geometry[1])
            button.clicked.connect(self.next)
        elif type == 'fin':
            button = QPushButton(title, self)
            button.move(geometry[0], geometry[1])
            button.clicked.connect(self.fin)
        elif type == 'houkai':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'mabataki':
            sld = QSlider(Qt.Horizontal, self)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.setGeometry(geometry[0], geometry[1], 60, 10)
        elif type == 'random':
            button = QPushButton(title, self)
            button.move(geometry[0], geometry[1])
            button.clicked.connect(self.random)
        elif type == 'torikesi':
            button = QPushButton(title, self)
            button.move(geometry[0], geometry[1])
            button.clicked.connect(self.torikesi)
        elif type == 'save':
            button = QPushButton(title, self)
            button.move(geometry[0], geometry[1])
            button.clicked.connect(self.save)

        if button is None:
            return sld
        else:
            return button


    def random(self):
        image_rand = self.gen.xp.random.randn(1, 512, 1, 1).astype('f')
        x = self.gen(image_rand, alpha=1.0)
        x = chainer.cuda.to_cpu(x.data)
        img = x[0].copy()
        utils.save_image(img, 'temp.jpg')
        _img = Image.open('temp.jpg')
        self.img = np.asarray(_img)
        # Create widget
        self.update_Figure(self.img)
        self.csv_file = [[0 for j in range(csv_len)]]

    def torikesi(self):
        self.csv_file[0][0] = 0
        self.csv_file[0][1] = 0
        self.csv_file[0][2] = 0
        self.csv_file[0][3] = 0
        self.csv_file[0][4] = 0
        self.csv_file[0][5] = 0
        self.csv_file[0][6] = 0
        self.csv_file[0][7] = 0
        self.csv_file[0][8] = 0
        self.csv_file[0][9] = 0
        self.csv_file[0][10] = 0
        self.csv_file[0][11] = 0
        self.csv_file[0][12] = 0

    def next(self):
        self.csv_file[0][0] = self.ude_migi_age.value()
        self.csv_file[0][1] = self.ude_hidari_age.value()
        self.csv_file[0][2] = self.ude_migi_tyuu.value()
        self.csv_file[0][3] = self.ude_hidari_tyuu.value()
        self.csv_file[0][4] = self.ude_migi_sita.value()
        self.csv_file[0][5] = self.ude_hidari_sita.value()
        self.csv_file[0][6] = self.kao_migi_muki.value()
        self.csv_file[0][7] = self.kao_hidari_muki.value()
        self.csv_file[0][8] = self.kao_ue_muki.value()
        self.csv_file[0][9] = self.kao_shita_muki.value()
        self.csv_file[0][10] = self.kao_shoumen_muki.value()
        self.csv_file[0][11] = self.houkai.value()
        self.csv_file[0][12] = self.mabataki.value()

        image_rand = np.array(self.vec2rand_model.predict_on_batch([self.csv_file]))
        image_rand =np.reshape(image_rand, (1,512,1,1))
        x = self.gen(image_rand, alpha=1.0)
        x = chainer.cuda.to_cpu(x.data)
        img = x[0].copy()
        utils.save_image(img, 'temp.jpg')
        _img = Image.open('temp.jpg')
        self.img = np.asarray(_img)
        # Create widget
        self.update_Figure(self.img)
        self.csv_file = [[0 for j in range(csv_len)]]

    def fin(self):
        rand = np.array(self.csv_file)
        paths = glob.glob('*.csv')
        np.savetxt('out_'+str(len(paths))+'.csv', rand, delimiter=',')
        sys.exit(app.exec_())

    def save(self):
        _img = Image.open('temp.jpg')
        _img.save('gen_img/gen_'+str(len(glob.glob('*.csv')))+'.jpg')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
