import sys

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox

from MainWindow import Ui_MainWindow
from program import SpamPredictor

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)


def okClicked():
	msg = QMessageBox()
	msg.setWindowTitle("Pressed Ok")
	msg.setText("Webo!")
	msg.exec()
	
	

def closeapp():
	window.close()

def initializeWindow():
	
	window.setWindowIcon(QtGui.QIcon('icon.png'))
	window.pushButton_2.clicked.connect(okClicked)
	window.pushButton.clicked.connect(closeapp)
	window.pushButton_3.clicked.connect(getUrl)


def getUrl():
	url = window.lineEdit.text()
	print(url)
	spamPredictor.loadComments(url)




app = QtWidgets.QApplication(sys.argv)
#Cogemos la clase mainwindow, implementada desde Qt desginer como ventana.
window = MainWindow()
window.show()

spamPredictor = SpamPredictor

initializeWindow()
app.exec()
