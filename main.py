from cluster import Clustring
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton
import sys


class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.clust = Clustring()

        self.setGeometry(300, 300, 600, 400)
        self.setFixedSize(600, 400)
        self.set_window()
        self.setWindowTitle("Text Classification")
        self.setWindowIcon(QtGui.QIcon('data/icon.png'))
        self.show()

    def set_window(self):
        self.text_BOX = QPlainTextEdit(self)
        self.text_BOX.setGeometry(0, 0, 600, 350)
        self.text_BOX.setPlaceholderText("Enter The Text Here ......")
        self.btn = QPushButton("Classify", self)
        self.btn.move(480, 360)
        self.btn.clicked.connect(self.clickedd)

    def clickedd(self):
        txt = self.text_BOX.toPlainText()
        msg = self.clust.predict(txt)
        QMessageBox.information(self, "classification", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
