import sys
from PyQt5.QtWidgets import QHBoxLayout, QFileDialog, QTextEdit, QCheckBox, QRadioButton, QLabel, QLineEdit, QSlider, QPushButton, QVBoxLayout, QApplication, QWidget
from PyQt5.QtWidgets import QMainWindow, QAction, qApp

class MenuDemo(QMainWindow):
    def __init__(self):
        super().__init__()

        #Create Menu Bar
        bar = self.menuBar()
        #Create Root Menus
        fileb = bar.addMenu('File')
        edit = bar.addMenu('Edit')
        #Creat Actions for Menus
        save_action = QAction('Save', self)
        save_action.setShortcut('Ctrl+S')

        new_action = QAction('New', self)
        new_action.setShortcut('Ctrl+N') #can't change to other string.

        quit_action = QAction('Quit', self)
        quit_action.setShortcut('Ctrl+Q')

        find_action = QAction('Find...', self)
        #save_action.setShortcut('Ctrl+S')

        replace_action = QAction('Replace...', self)
        #save_action.setShortcut('Ctrl+S')

        #add actions to menus
        fileb.addAction(new_action)
        fileb.addAction(save_action)
        fileb.addAction(quit_action)
        find_menu = edit.addMenu('Find')
        find_menu.addAction(find_action)
        find_menu.addAction(replace_action)
        #events
        quit_action.triggered.connect(self.quit_trigger)
        fileb.triggered.connect(self.selected)


        self.setWindowTitle("My Menus")
        self.resize(600,400)

        self.show()

    def quit_trigger(self):
        qApp.quit()

    def selected(self, q):  #why q? how this q works?
        print(q.text() + ' selected')

app = QApplication(sys.argv)
menus = MenuDemo()
sys.exit(app.exec_())
