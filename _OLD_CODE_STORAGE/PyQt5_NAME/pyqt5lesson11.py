import sys
#from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QTextEdit, QCheckBox, QRadioButton, QLabel, QLineEdit, QSlider, QPushButton, QVBoxLayout, QApplication, QWidget)
from PyQt5.QtCore import Qt

class Notepad(QWidget):
    def __init__(self):

        super(Notepad, self).__init__()
        self.text = QTextEdit(self)
        self.clr_btn = QPushButton('clear')
        self.sv_btn = QPushButton('save')

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(self.text)
        layout.addWidget(self.clr_btn)
        layout.addWidget(self.sv_btn)
        self.clr_btn.clicked.connect(self.clear_text)
        self.sv_btn.clicked.connect(self.save_text)

        self.setLayout(layout)
        self.setWindowTitle('PyQt5 lesson 11')

        self.show()
    def clear_text(self):
        self.text.clear()
    def save_text(self):
        with open('test.txt', 'w') as f:
            my_text = self.text.toPlainText()
            f.write(my_text)

app = QApplication(sys.argv)
writer = Notepad()
sys.exit(app.exec_())
