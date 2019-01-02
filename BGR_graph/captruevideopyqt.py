import cv2
import numpy as np
from PyQt5.QtWidgets import qApp, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtWidgets import QApplication


class Window(QWidget):
    def __init__(self):

        c = cv2.VideoCapture(0)

        QWidget.__init__(self)
        self.setWindowTitle('Control Panel')

        self.start_button = QPushButton('Start',self)
        self.start_button.clicked.connect(lambda : self.startCapture(c))

        self.end_button = QPushButton('End',self)
        self.end_button.clicked.connect(self.endCapture)

        self.quit_button = QPushButton('Quit',self)
        self.quit_button.clicked.connect(lambda : self.quit(c))

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)

        self.setLayout(vbox)
        self.setGeometry(100,100,200,200)
        self.show()

    def startCapture(self, cap):
        print ("pressed start")
        while(True):
            ret, frame = cap.read()
            cv2.imshow("Capture", frame)
            cv2.waitKey(20)

    def endCapture(self):
        print ("pressed End")
        cv2.destroyAllWindows()

    def quitCapture(self, cap):
        print ("pressed Quit")
        #cv2.destroyAllWindows()
        #cap.release()
        qApp.quit()

if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
