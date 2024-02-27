from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

class Button(QToolButton):
    def __init__(self, text, clicked):
        super().__init__()  # 부모 클래스의 생성자 호출
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # 버튼의 크기 정책 설정
        self.setText(text)  # 버튼에 텍스트 설정
        self.clicked.connect(clicked)  # 버튼 클릭 시그널과 연결할 함수 지정

class App(QWidget):
    def __init__(self): 
        super().__init__()  # 부모 클래스의 생성자 호출
        
        # 윈도우 제목 및 크기 설정
        self.title = "계산기"
        self.setWindowTitle(self.title)
        self.left = 100
        self.top = 200
        self.width = 350
        self.height = 500
        self.setGeometry(self.left, self.top, self.width, self.height)

        # 디스플레이 위젯 설정
        self.display = QLineEdit("")  # 초기값 설정
        self.display.setReadOnly(True)  # 읽기 전용으로 설정
        self.display.setAlignment(Qt.AlignRight)  # 텍스트를 오른쪽 정렬
        self.display.setStyleSheet("border:0px; font-size:20pt; font-family:Nanum Gothic; font-weight:bold; padding:10px")  # 스타일 지정

        # 그리드 레이아웃 및 수직 박스 레이아웃 설정
        gridLayout = QGridLayout()
        gridLayout.setSizeConstraint(QLayout.SetFixedSize)
        layout = QVBoxLayout(self)
        layout.addWidget(self.display)
        layout.addLayout(gridLayout)

        # 버튼 생성 및 배치
        self.createButtons(gridLayout)

    def createButtons(self, gridLayout):
        # 버튼 텍스트와 클릭 시 실행될 함수 지정
        buttons = [
            ("CE", self.clear),
            ("C", self.clearAll),
            ("Back", self.backDelete),
            ("/", self.clickButtons),
            ("7", self.clickButtons),
            ("8", self.clickButtons),
            ("9", self.clickButtons),
            ("*", self.clickButtons),
            ("4", self.clickButtons),
            ("5", self.clickButtons),
            ("6", self.clickButtons),
            ("-", self.clickButtons),
            ("1", self.clickButtons),
            ("2", self.clickButtons),
            ("3", self.clickButtons),
            ("+", self.clickButtons),
            ("R", self.reverse),
            ("0", self.clickButtons),
            (".", self.clickButtons),
            ("=", self.equals),
        ]

        # 버튼의 위치 설정
        positions = [(i, j) for i in range(5) for j in range(4)]
        
        # 버튼 생성 및 그리드 레이아웃에 추가
        for (text, clicked), position in zip(buttons, positions):
            button = Button(text, clicked)
            gridLayout.addWidget(button, *position)

    # 버튼 클릭 시 실행될 함수들
    def clear(self):
        self.display.clear()

    def clearAll(self):
        self.display.setText("")

    def backDelete(self):
        self.display.backspace()

    def clickButtons(self):
        button = self.sender()
        self.display.insert(button.text())

    def equals(self):
        try:
            result = eval(self.display.text())
            self.display.setText(str(result))
        except Exception as e:
            self.display.setText("Error")

    def reverse(self):
        text = self.display.text()
        if text[0] == '-':
            self.display.setText(text[1:])
        else:
            self.display.setText('-' + text)

# 메인 실행 코드
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    calc = App()
    calc.show()
    sys.exit(app.exec_())
