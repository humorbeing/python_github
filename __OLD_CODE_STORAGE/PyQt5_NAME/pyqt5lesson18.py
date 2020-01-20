import sys
import os
import csv
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

    def open_sheet(self):
        path = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('HOME'), 'CSV(*.csv)') # filter, only show csv files
        if path[0] != '':
            with open(path[0], newline = '') as csv_file:
                self.setRowCount(0)
                self.setColumnCount(10)
                my_file = csv.reader(csv_file, delimiter = ',', quotechar= '|')
                for row_data in my_file:
                    row = self.rowCount()
                    self.insertRow(row)
                    if len(row_data) > 10:
                        self.setColumnCount(len(row_data))
                    for column, stuff in enumerate(row_data):
                        item = QTableWidgetItem(stuff)
                        self.setItem(row, column, item)


class Sheet(QMainWindow):
    def __init__(self):
        super().__init__()

        self.form_widget = MyTable(10,10)
        self.setCentralWidget(self.form_widget)
        col_headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J' ]
        self.form_widget.setHorizontalHeaderLabels(col_headers)

        self.form_widget.open_sheet()

        self.show()

app = QApplication(sys.argv)
sheet = Sheet()
sys.exit(app.exec_())