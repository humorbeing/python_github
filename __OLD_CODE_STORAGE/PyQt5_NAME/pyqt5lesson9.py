import sys
#from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QCheckBox, QLabel, QLineEdit, QSlider, QPushButton, QVBoxLayout, QApplication, QWidget)
from PyQt5.QtCore import Qt

class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.lbl = QLabel()
        self.chx = QCheckBox('Do you like dogs?')
        self.btn = QPushButton('push me')

        v_box = QVBoxLayout()
        v_box.addWidget(self.lbl)
        v_box.addWidget(self.chx)
        v_box.addWidget(self.btn)


        self.setLayout(v_box)

        self.setWindowTitle('PyQt5 lesson 9')

        self.btn.clicked.connect(lambda: self.btn_clk(self.chx.isChecked(), self.lbl))

        self.show()

    def btn_clk(self, chk, lbl):
        if chk:
            lbl.setText('i guess you like dogs')
        else:
            lbl.setText('Dog hater then')


app = QApplication(sys.argv)
a_window = Window()
sys.exit(app.exec_())
