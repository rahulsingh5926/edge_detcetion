import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class EdgeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Edge Detection and Display Application')
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        btn_open = QPushButton('Open Image', self)
        btn_open.clicked.connect(self.open_image)

        layout = QVBoxLayout()
        layout.addWidget(btn_open)
        layout.addWidget(self.label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def open_image(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)",
                                                  options=options)
        if filePath:
            self.process_image(filePath)

    def process_image(self, filePath):
        image = cv2.imread(filePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 90)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and take the largest contour
        filtered_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        if not filtered_contours:
            return  # No significant contours found
        
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        
        # Create a mask to represent the area inside the contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Overlay the detected region on the original image
        result = image.copy()
        result[mask == 0] = [0, 0, 0]  # Set areas outside the mask to black
        
        qImg = QImage(result.data, result.shape[1], result.shape[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        pixmap_resized = pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap_resized)

def main():
    app = QApplication(sys.argv)
    window = EdgeDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
