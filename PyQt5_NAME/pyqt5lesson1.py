import sys
from PyQt5 import QtWidgets

def window():
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QWidget()
    w.setWindowTitle('PyQt5 Lesson 1')
    w.setGeometry(100,100,600,400)
    w.show()
    sys.exit(app.exec_())

window()
