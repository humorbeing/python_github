import sys
from PyQt5 import QtWidgets

def window():
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QWidget()
    b = QtWidgets.QPushButton('push me')
    l = QtWidgets.QLabel('look at me')
    h_box = QtWidgets.QHBoxLayout()# left right
    h_box.addStretch()
    h_box.addWidget(l)
    h_box.addStretch()


    v_box = QtWidgets.QVBoxLayout()# up down
    v_box.addStretch()
    v_box.addWidget(b)
    v_box.addStretch()
    v_box.addLayout(h_box)
    v_box.addStretch()

    
    w.setLayout(v_box)

    w.setWindowTitle('PyQt5 Lesson 1')

    w.show()
    sys.exit(app.exec_())

window()
