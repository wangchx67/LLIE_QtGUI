import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout,QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap,QImage,QColor,QPainter,QIcon
from PIL import Image, ImageOps,ImageEnhance
import numpy as np
import time
from archs.color_utils.color_transfer import color_transfer
from define_models import process
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_histogram(image):
    histogram = image.histogram()
    histogram = np.array(histogram)
    return histogram

def draw_histogram(histogram):
    width, height = 128, 80
    histogram_image = QImage(width, height, QImage.Format_RGB32)
    histogram_image.fill(QColor(255, 255, 255))

    painter = QPainter(histogram_image)
    painter.setPen(QColor(0, 0, 0))

    max_value = max(histogram)
    for i in range(256):
        x = i//2
        y = height - histogram[i] * height / max_value
        painter.drawLine(x, height, x, y)

    painter.end()
    return histogram_image

def PIL2QPix(pil):
    image_data = pil.tobytes('raw', pil.mode)
    qimage = QImage(image_data, pil.width, pil.height, pil.width * 3,
                    QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 loadUi 加载 UI 文件
        self.ui = loadUi('./bishe.ui')  # 请将文件名替换为您的 UI 文件的路径
        self.ui.resetButton.clicked.connect(self.reset)
        self.ui.readButton.clicked.connect(self.loadImage)
        self.ui.readButton.setIcon(QIcon.fromTheme('SP_TitleBarCloseButton'))
        self.ui.runButton.clicked.connect(self.enhance)
        self.ui.gamma_radioButton.clicked.connect(self.ifcustomized)
        self.ui.he_radioButton.clicked.connect(self.ifcustomized)
        self.ui.FourLLIE_radioButton.clicked.connect(self.ifcustomized)
        self.ui.CCNet_radioButton.clicked.connect(self.ifcustomized)
        self.ui.loadref.clicked.connect(self.loadrefImage)
        self.input_image = None
        self.output_image = None
        self.ref_image = None
        self.pre_ref1 = Image.open('./ui_imgs/refs/ref_5k1.png')
        self.pre_ref2 = Image.open('./ui_imgs/refs/ref_5k2.png')
        self.pre_ref3 = Image.open('./ui_imgs/refs/ref_5k5.png')
        self.pre_ref4 = Image.open('./ui_imgs/refs/ref_5k8.png')
        self.pre_ref5 = Image.open('./ui_imgs/refs/ref_5k11.png')

        self.ui.gammaSlider.setSingleStep(1)
        self.ui.gammaSlider.setRange(10, 30)
        self.ui.gammaSlider.setValue(15)

        self.ui.saturationSlider.setSingleStep(1)
        self.ui.saturationSlider.setRange(10, 30)
        self.ui.saturationSlider.setValue(10)

        self.initUI()
    def reset(self):
        self.ui.inputImg.clear()
        self.ui.enhancedImg.clear()
        self.ui.referneceImg.clear()
        self.ui.hist_pre.clear()
        self.ui.hist_post.clear()
        self.ui.PSNR.clear()
        self.ui.SSIM.clear()
        self.ui.times.clear()

    def loadImage(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '选择图片文件', '', 'Image Files (*.png *.jpg *.bmp)')
        # 如果选择了文件，则加载图片并显示在标签中
        if file_name:
            pixmap = QPixmap(file_name)
            self.ui.inputImg.setPixmap(pixmap.scaled(self.ui.inputImg.size()))
            self.input_image = Image.open(file_name).resize((321, 231))
            self.target_image = Image.open(file_name.replace('inputs','targets')).resize((321, 231))

    def loadrefImage(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '选择图片文件', '', 'Image Files (*.png *.jpg *.bmp)')
        # 如果选择了文件，则加载图片并显示在标签中
        if file_name:
            pixmap = QPixmap(file_name)
            self.ui.referneceImg.setPixmap(pixmap.scaled(self.ui.referneceImg.size()))
            self.ref_image = color_transfer(np.asarray(Image.open(file_name).resize((321, 231))),np.asarray(self.input_image), clip=True, preserve_paper=True)

    def enhance(self):
        if self.ui.gamma_radioButton.isChecked():
            self.enhance_('gamma')
        elif self.ui.he_radioButton.isChecked():
            self.enhance_('he')
        elif self.ui.FourLLIE_radioButton.isChecked():
            self.enhance_('FourLLIE')
        elif self.ui.CCNet_radioButton.isChecked():
            self.enhance_('CCNet')
        self.ui.enhancedImg.setPixmap(PIL2QPix(self.output_image))
        self.evaluate()

    def initUI(self):
        # 设置窗口的布局
        layout = QVBoxLayout()
        layout.addWidget(self.ui)
        self.setLayout(layout)
        if not self.ui.CCNet_radioButton.isChecked():
            self.ui.customizedGroup.setEnabled(False)
        self.ui.style1Button.setIcon(QIcon(PIL2QPix(self.pre_ref1).scaled(self.ui.style1Button.size())))
        self.ui.style1Button.clicked.connect(lambda e: self.setref_pre(self.pre_ref1))
        self.ui.style2Button.setIcon(QIcon(PIL2QPix(self.pre_ref2).scaled(self.ui.style1Button.size())))
        self.ui.style2Button.clicked.connect(lambda e: self.setref_pre(self.pre_ref2))
        self.ui.style3Button.setIcon(QIcon(PIL2QPix(self.pre_ref3).scaled(self.ui.style1Button.size())))
        self.ui.style3Button.clicked.connect(lambda e: self.setref_pre(self.pre_ref3))
        self.ui.style4Button.setIcon(QIcon(PIL2QPix(self.pre_ref4).scaled(self.ui.style1Button.size())))
        self.ui.style4Button.clicked.connect(lambda e: self.setref_pre(self.pre_ref4))
        self.ui.style5Button.setIcon(QIcon(PIL2QPix(self.pre_ref5).scaled(self.ui.style1Button.size())))
        self.ui.style5Button.clicked.connect(lambda e: self.setref_pre(self.pre_ref5))

        self.setWindowTitle('低光照图像增强及个性化增强分析软件')

    def enhance_(self,type):
        if type=='gamma':
            t1 = time.time()
            enhancer = ImageEnhance.Contrast(self.input_image)
            gamma_value = self.ui.gammaSlider.value()/10
            t2 = time.time()
            self.t = t2-t1
            self.output_image = enhancer.enhance(gamma_value)
        elif type == 'he':
            t1 = time.time()
            r, g, b = self.input_image.split()
            r_equalized = ImageOps.equalize(r)
            g_equalized = ImageOps.equalize(g)
            b_equalized = ImageOps.equalize(b)
            t2 = time.time()
            self.t = t2 - t1
            self.output_image = Image.merge('RGB', (r_equalized, g_equalized, b_equalized))
        elif type=='FourLLIE':
            self.output_image, self.t = process('FourLLIE',self.input_image,data = 'lol2real')
        elif type=='CCNet':
            self.output_image, self.t = process('CCNet',self.input_image,self.ref_image,self.ui.saturationSlider.value()/10,data = '5k')
    
    def evaluate(self):
        histogram_input = calculate_histogram(self.input_image)
        histogram_ouput = calculate_histogram(self.output_image)
        histogram_image_input = draw_histogram(histogram_input)
        histogram_image_ouput = draw_histogram(histogram_ouput)
        pixmap_input = QPixmap.fromImage(histogram_image_input)
        self.ui.hist_pre.setPixmap(pixmap_input)
        pixmap_ouput = QPixmap.fromImage(histogram_image_ouput)
        self.ui.hist_post.setPixmap(pixmap_ouput)
        arr1 = np.asarray(self.output_image)
        arr2 = np.asarray(self.target_image)
        p = psnr(arr1,arr2)
        s = ssim(arr1,arr2,channel_axis=2,data_range=255)
        self.ui.PSNR.setText(str(p)+'dB')
        self.ui.SSIM.setText(str(s))
        self.ui.times.setText(str(self.t)+'s')

    def ifcustomized(self):
        if self.ui.gamma_radioButton.isChecked():
            self.ui.customizedGroup.setEnabled(False)
        elif self.ui.he_radioButton.isChecked():
            self.ui.customizedGroup.setEnabled(False)
        elif self.ui.FourLLIE_radioButton.isChecked():
            self.ui.customizedGroup.setEnabled(False)
        elif self.ui.CCNet_radioButton.isChecked():
            self.ui.customizedGroup.setEnabled(True)
    def setref_pre(self,img):
        # self.ref_image = img
        self.ref_image = color_transfer(np.asarray(img.resize((321, 231))),
                                        np.asarray(self.input_image), clip=True, preserve_paper=True)
        self.ui.referneceImg.setPixmap(PIL2QPix(img).scaled(self.ui.referneceImg.size()))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())


