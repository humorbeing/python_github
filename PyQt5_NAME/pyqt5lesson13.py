import sys
from PyQt5.QtWidgets import QHBoxLayout, QFileDialog, QTextEdit, QCheckBox, QRadioButton, QLabel, QLineEdit, QSlider, QPushButton, QVBoxLayout, QApplication, QWidget
import os

class Notepad(QWidget):
    def __init__(self):
        super(Notepad, self).__init__()
        self.text = QTextEdit(self)
        self.clr_btn = QPushButton('clear')
        self.sv_btn = QPushButton('save')
        self.opn_btn = QPushButton('open')

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        h_layout = QHBoxLayout()

        h_layout.addWidget(self.clr_btn)
        h_layout.addWidget(self.sv_btn)
        h_layout.addWidget(self.opn_btn)

        layout.addWidget(self.text)
        layout.addLayout(h_layout) #!!! addlayout,not addwidget

        self.clr_btn.clicked.connect(self.clear_text)
        self.sv_btn.clicked.connect(self.save_text)
        self.opn_btn.clicked.connect(self.open_text)

        self.setLayout(layout)
        self.setWindowTitle('PyQt5 lesson 13')

        self.show()

    def clear_text(self):
        self.text.clear()
    def save_text(self):
        filename = QFileDialog.getSaveFileName(self, 'Save File', os.getenv('HOME'))
        with open(filename[0], 'w') as f:
            my_text = self.text.toPlainText()
            f.write(my_text)

    def open_text(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File', os.getenv('HOME'))
        with open(filename[0], 'r') as f:
            file_text = f.read()
            self.text.setText(file_text)


app = QApplication(sys.argv)
writer = Notepad()
sys.exit(app.exec_())
