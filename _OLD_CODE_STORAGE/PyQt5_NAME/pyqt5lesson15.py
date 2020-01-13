import sys
from PyQt5.QtWidgets import QHBoxLayout, QFileDialog, QTextEdit, QCheckBox, QRadioButton, QLabel, QLineEdit, QSlider, QPushButton, QVBoxLayout, QApplication, QWidget
import os
from PyQt5.QtWidgets import QMainWindow, QAction, qApp

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

class Writer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.form_widget = Notepad()
        self.setCentralWidget(self.form_widget)

        self.init_ui()

    def init_ui(self):

        bar = self.menuBar()
        #Create Root Menus
        fileb = bar.addMenu('File')
        save_action = QAction('&Save', self)
        save_action.setShortcut('Ctrl+S')

        new_action = QAction('New', self)
        new_action.setShortcut('Ctrl+N') #can't change to other string.

        open_action = QAction('&Open', self)
        open_action.setShortcut('Ctrl+O')

        quit_action = QAction('&Quit', self)
        quit_action.setShortcut('Alt+Q')

        fileb.addAction(new_action)
        fileb.addAction(save_action)
        fileb.addAction(open_action)
        fileb.addAction(quit_action)

        quit_action.triggered.connect(self.quit_trigger)
        fileb.triggered.connect(self.respond)

        self.show()

    def quit_trigger(self):
        qApp.quit()

    def respond(self,q):
        signal = q.text()
        if signal == 'New':
            self.form_widget.clear_text()
        elif signal == '&Open':
            self.form_widget.open_text()
        elif signal == '&Save':
            self.form_widget.save_text()


app = QApplication(sys.argv)
writer = Writer()
sys.exit(app.exec_())
