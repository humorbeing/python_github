import sys
from PyQt5.QtWidgets import QHBoxLayout, QFileDialog, QCheckBox
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QTextEdit
from PyQt5.QtWidgets import QRadioButton, QLabel, QLineEdit
from PyQt5.QtWidgets import QSlider, QPushButton, QVBoxLayout
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidget

class MyTable(QTableWidget):
    def __init__(self, r, c):
        super().__init__(r, c)

        self.show()
class Sheet(QMainWindow):
    def __init__(self):
        super().__init__()

        self.from_widget = MyTable(10,10)
        self.setCentralWidget(self.from_widget)

        self.show()

app = QApplication(sys.argv)
#table = MyTable(10,10)
sheet = Sheet()
sys.exit(app.exec_())
