import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('My First Application')
        self.move(300,0)

        label1 = QLabel('aaa',self)
        label1.setAlignment(Qt.AlignCenter)
        label1.move(0,10)

        label2 = QLabel(self)
        label2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label2.setGeometry(100, 200, 500, 550)
        pixmap = QPixmap('pic1.png')

        label2.move(0,30)
        label2.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        self.setGeometry(100, 200, 500, 550)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    print(len(sys.argv),sys.argv[0])
    ex = MyApp()
    sys.exit(app.exec_())