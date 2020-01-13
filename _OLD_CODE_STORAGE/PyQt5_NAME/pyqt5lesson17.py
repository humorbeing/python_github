import sys
from PyQt5.QtWidgets import QHBoxLayout, QFileDialog, QCheckBox
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QTextEdit
from PyQt5.QtWidgets import QRadioButton, QLabel, QLineEdit
from PyQt5.QtWidgets import QSlider, QPushButton, QVBoxLayout
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidget
from PyQt5.QtWidgets import QTableWidgetItem

class MyTable(QTableWidget):
    def __init__(self, r, c):
        super().__init__(r, c)
        self.init_ui()

    def init_ui(self):
        self.cellChanged.connect(self.c_current)
        self.show()

    def c_current(self):
        row = self.currentRow()
        col = self.currentColumn()
        value = self.item(row, col)
        value = value.text()
        print("The current cell is ", row, ", ", col)
        print("In this cell we have ", value)

class Sheet(QMainWindow):
    def __init__(self):
        super().__init__()

        self.from_widget = MyTable(10,10)
        self.setCentralWidget(self.from_widget)
        col_headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J' ]
        self.from_widget.setHorizontalHeaderLabels(col_headers)

        number = QTableWidgetItem('10')
        self.from_widget.setCurrentCell(1,1) #this triggers c_current funtion. better have this before any edition.
        self.from_widget.setItem(1,1, number)

        self.show()

app = QApplication(sys.argv)
sheet = Sheet()
sys.exit(app.exec_())
