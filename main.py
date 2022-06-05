from queue import Empty
import sys

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox

from MainWindow import Ui_MainWindow
from program import SpamPredictor

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

global spammers

def okClicked():

	global spammers
	if("spammers" in globals()):

		if(len(spammers) != 0):
			msg = QMessageBox()
			msg.setWindowIcon(QtGui.QIcon('save.png'))
			msg.setWindowTitle("Saved")
			msg.setText("Results Saved succesfully in file -> potentialSpammers.txt")
			msg.exec()
			with open('potentialSpammers.txt','w') as f:
				for line in spammers.keys():
					f.write(line)
					f.write('\n')
			f.close()
		
	

def closeapp():
	window.close()

def initializeWindow():
	
	window.setWindowIcon(QtGui.QIcon('icon.png'))
	window.pushButton_2.clicked.connect(okClicked)
	window.pushButton.clicked.connect(closeapp)
	window.pushButton_3.clicked.connect(getUrl)
	print("Window initialized")




def getUrl():
	global spammers
	url = window.lineEdit.text()
	

	if(url == ""):
		url = "https://www.youtube.com/watch?v=hD1YtmKXNb4"

	print(url)
	
	if(url != ""):
		spamPredictor.loadComments(url, window.spinBox.value())
		spammers = spamPredictor.getSuspectedSpammers()
		print("Spammers succesfully identyfied")
		window.tableWidget.setRowCount(spammers.__len__())

		i = 0
		
		for key,element in spammers.items():
			
			window.tableWidget.setItem(i,0, QtWidgets.QTableWidgetItem(key))
			window.tableWidget.setItem(i,1, QtWidgets.QTableWidgetItem(element))
			i+=1
	




app = QtWidgets.QApplication(sys.argv)
#Cogemos la clase mainwindow, implementada desde Qt desginer como ventana.
window = MainWindow()
spamPredictor = SpamPredictor()

window.show()
initializeWindow()

app.exec()
