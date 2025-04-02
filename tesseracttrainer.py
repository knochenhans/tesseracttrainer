import os
from typing import List
from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QMainWindow,
    QWidget,
)

from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal, QSettings
import sys

from model_trainer import ModelTrainer


class CustomTextEditor(QLineEdit):
    ctrlEnterPressed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setPlaceholderText("Enter text here...")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("font-size: 20px; font-family: 'Courier New', monospace;")

    def keyPressEvent(self, event):
        if (
            event.key() == Qt.Key.Key_Return
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            self.ctrlEnterPressed.emit()
        else:
            super().keyPressEvent(event)


class ImageTextWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image and Text Viewer")
        self.resize(800, 600)  # Set a default size
        self.showMaximized()

        self.settings = QSettings("TesseractTrainer", "TesseractTrainer")
        self.base_path: str = str(self.settings.value("base_path", ""))

        # Central widget
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

        # Image label
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        layout.addWidget(self.image_label, stretch=1)

        # Grid layout for buttons and path label
        grid_layout = QVBoxLayout()

        # Text editor
        self.text_editor = CustomTextEditor(self)
        self.text_editor.ctrlEnterPressed.connect(self.load_next)
        grid_layout.addWidget(self.text_editor)

        # Previous, Next, Remove, Train buttons
        button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.load_previous)
        button_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.load_next)
        button_layout.addWidget(self.next_button)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_current_pair)
        button_layout.addWidget(self.remove_button)

        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.train_model)
        button_layout.addWidget(self.train_button)

        grid_layout.addLayout(button_layout)

        # Second row: Path button and label
        path_layout = QHBoxLayout()
        self.set_path_button = QPushButton("Set GT Base Path")
        self.set_path_button.clicked.connect(self.set_ground_truth_base_path)
        path_layout.addWidget(self.set_path_button)

        self.path_label = QLabel(f"Base Path: {self.base_path}")
        path_layout.addWidget(self.path_label)

        grid_layout.addLayout(path_layout)

        layout.addLayout(grid_layout)

        self.file_base_names = self.get_file_base_names()
        self.current_index = 0

        if self.file_base_names:
            self.load_pair(self.file_base_names[self.current_index])

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def train_model(self) -> None:
        model_trainer = ModelTrainer(
            model_name="deu",
            base_dir=self.base_path,
            tessdata_dir="/usr/share/tessdata",
        )
        model_trainer.train_model()

    def set_ground_truth_base_path(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select Ground Thruth Base Path"
        )
        if directory:
            self.base_path = directory
            self.settings.setValue("base_path", self.base_path)
            self.file_base_names = self.get_file_base_names()
            self.current_index = 0
            if self.file_base_names:
                self.load_pair(self.file_base_names[self.current_index])
                self.path_label.setText(f"Base Path: {self.base_path}")

    def get_file_base_names(self) -> List[str]:
        file_base_paths = []
        if self.base_path:
            for file_name in os.listdir(self.base_path):
                if file_name.endswith(".tif"):
                    base_name = os.path.splitext(file_name)[0]
                    gt_file = os.path.join(self.base_path, f"{base_name}.gt.txt")
                    if os.path.exists(gt_file):
                        file_base_paths.append(os.path.join(self.base_path, base_name))
                    else:
                        raise FileNotFoundError(
                            f"Ground truth file not found for {file_name}"
                        )
                elif file_name.endswith(".gt.txt"):
                    base_name = os.path.splitext(file_name)[0]
                    base_name = base_name.replace(".gt", "")
                    image_file = os.path.join(self.base_path, f"{base_name}.tif")
                    if not os.path.exists(image_file):
                        raise FileNotFoundError(f"Image file not found for {file_name}")
        return file_base_paths

    def load_image(self, file_path: str) -> None:
        if file_path:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    pixmap.width() * 2,
                    pixmap.height() * 2,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("Failed to load image")

    def load_text(self, file_path: str) -> None:
        if file_path:
            with open(file_path, "r") as file:
                text = file.read()
                self.text_editor.setText(text)
        else:
            raise ValueError(f"File path is empty: {file_path}")

    def load_pair(self, file_base_name: str) -> None:
        if file_base_name and not self.is_removed_pair(file_base_name):
            base_name = os.path.splitext(file_base_name)[0]
            image_file = f"{base_name}.tif"
            gt_file = f"{base_name}.gt.txt"

            if os.path.exists(image_file):
                self.load_image(image_file)
            else:
                raise FileNotFoundError(f"Image file not found for {file_base_name}")

            if os.path.exists(gt_file):
                self.load_text(gt_file)
            else:
                raise FileNotFoundError(
                    f"Ground truth file not found for {file_base_name}"
                )

            self.setWindowTitle(os.path.basename(image_file))

    def save_text(self, file_path: str) -> None:
        if file_path:
            with open(file_path, "w") as file:
                text = self.text_editor.text()
                file.write(text)
        else:
            raise ValueError(f"File path is empty: {file_path}")

    def load_previous(self) -> None:
        self.fetch_current_pair(-1)

    def load_next(self) -> None:
        self.fetch_current_pair(1)

    def fetch_current_pair(self, step: int) -> None:
        self.save_text(f"{self.file_base_names[self.current_index]}.gt.txt")

        new_index = self.current_index + step
        if 0 <= new_index < len(self.file_base_names):
            self.current_index = new_index
            self.load_pair(self.file_base_names[self.current_index])
            self.text_editor.setFocus()

    def remove_current_pair(self) -> None:
        if self.file_base_names:
            file_base_name = self.file_base_names[self.current_index]
            # Remove the files
            os.remove(f"{file_base_name}.tif")
            os.remove(f"{file_base_name}.gt.txt")

            # Log the removed pair
            self.log_removed_pair(file_base_name)

            # Update the list and load the next pair
            del self.file_base_names[self.current_index]
            if self.file_base_names:
                self.load_pair(self.file_base_names[self.current_index])
            else:
                self.image_label.setText("No image loaded")
                self.text_editor.clear()

    def log_removed_pair(self, file_base_name: str) -> None:
        removed_pairs_file = os.path.join(self.base_path, "removed_pairs.txt")
        if os.path.exists(removed_pairs_file):
            with open(removed_pairs_file, "r") as log_file:
                logged_pairs = log_file.read().splitlines()
        else:
            logged_pairs = []

        if file_base_name not in logged_pairs:
            with open(removed_pairs_file, "a") as log_file:
                log_file.write(f"{file_base_name}\n")

    def is_removed_pair(self, file_base_name: str) -> bool:
        removed_pairs_file = os.path.join(self.base_path, "removed_pairs.txt")
        if os.path.exists(removed_pairs_file):
            with open(removed_pairs_file, "r") as log_file:
                logged_pairs = log_file.read().splitlines()
            return file_base_name in logged_pairs
        return False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageTextWindow()
    window.show()
    sys.exit(app.exec())
