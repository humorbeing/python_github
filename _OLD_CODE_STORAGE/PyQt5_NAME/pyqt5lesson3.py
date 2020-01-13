import sys
from PyQt5 import QtWidgets

def window():
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QWidget()
    b = QtWidgets.QPushButton(w)
    l = QtWidgets.QLabel(w)
    b.setText('push me')
    l.setText('look at me')
    w.setWindowTitle('PyQt5 Lesson 1')
    b.move(100,50)
    l.move(110,100)
    w.setGeometry(100,100,600,400)
    w.show()
    sys.exit(app.exec_())

window()
