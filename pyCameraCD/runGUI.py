import sys
from PyQt5 import QtWidgets
from MainFrame import MainFrame

if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    MainFrame = MainFrame()
    MainFrame.show()
    sys.exit(app.exec_())
