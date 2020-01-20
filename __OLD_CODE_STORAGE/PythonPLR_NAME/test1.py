from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys

class Window(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)
        self.setLayout(QVBoxLayout())
        #self.layout().addWidget(QLabel("<font color='red'>This is the text</font"))
        # let the whole window be a glass
        self.setAttribute(Qt.WA_TranslucentBackground)
        #self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        #self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        from ctypes import windll, c_int, byref
        windll.dwmapi.DwmExtendFrameIntoClientArea(c_int(self.winId()), byref(c_int(-1)))
        self.move(200, 200)
    def mousePressEvent(self, event):
        self.repaint()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    wnd = Window()
    wnd.show()
    sys.exit(app.exec_())
