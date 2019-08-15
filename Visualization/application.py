import sys
import qdarkstyle
from gui_main import MainWindow
from PyQt5.QtWidgets import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    desktopWidget = QApplication.desktop()
    screenRect = desktopWidget.screenGeometry()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainWindow(screenRect.width(),screenRect.height())
    window.setWindowTitle("model_view")
    window.show()
    sys.exit(app.exec_())
