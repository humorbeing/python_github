from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import sys


app = QApplication(sys.argv)

window = QMainWindow()
window.setAttribute(Qt.WA_TranslucentBackground)
#window.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
window.setWindowFlags(window.windowFlags() | Qt.WindowStaysOnTopHint)

#window.setAttribute(Qt.WA_NoSystemBackground, True)
#window.setAttribute(Qt.WA_TranslucentBackground, True)
#window.setWindowOpacity(0.5)

window.show()

sys.exit(app.exec_())
