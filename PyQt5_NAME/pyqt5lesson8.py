import sys
#from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QLineEdit, QSlider, QPushButton, QVBoxLayout, QApplication, QWidget)
from PyQt5.QtCore import Qt

class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.le = QLineEdit()
        self.b1 = QPushButton('Clear')
        self.b2 = QPushButton('Print')
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(1)
        self.sl.setMaximum(99)
        self.sl.setValue(25)
        self.sl.setTickInterval(10)
        self.sl.setTickPosition(QSlider.TicksBelow)

        v_box = QVBoxLayout()
        v_box.addWidget(self.le)
        v_box.addWidget(self.b1)
        v_box.addWidget(self.b2)
        v_box.addWidget(self.sl)

        self.setLayout(v_box)

        self.setWindowTitle('PyQt5 lesson 8')
        #lambda send value to funtion
        self.b1.clicked.connect(lambda:self.btn_clk(self.b1, 'hello from clear'))
        self.b2.clicked.connect(lambda:self.btn_clk(self.b2, 'hello from print'))
        self.sl.valueChanged.connect(self.v_change)



        self.show()

    def btn_clk(self, b, string):
        if b.text() == 'Print':
            print(self.le.text())
        else:
            self.le.clear()
        print(string)

    def v_change(self):
        my_value = str(self.sl.value())
        self.le.setText(my_value)

app = QApplication(sys.argv)
a_window = Window()
sys.exit(app.exec_())
