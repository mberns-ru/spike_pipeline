import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QHBoxLayout, QComboBox

class FolderInput(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create layout
        layout = QVBoxLayout()

        # Input folder selection
        self.inputFolderLabel = QLabel("Input Folder:")
        self.inputFolderPath = QLineEdit(self)
        self.inputFolderButton = QPushButton("Browse...")
        self.inputFolderButton.clicked.connect(self.browseInputFolder)
        inputFolderLayout = QHBoxLayout()
        inputFolderLayout.addWidget(self.inputFolderPath)
        inputFolderLayout.addWidget(self.inputFolderButton)

        # Output folder selection
        self.outputFolderLabel = QLabel("Output Folder:")
        self.outputFolderPath = QLineEdit(self)
        self.outputFolderButton = QPushButton("Browse...")
        self.outputFolderButton.clicked.connect(self.browseOutputFolder)
        outputFolderLayout = QHBoxLayout()
        outputFolderLayout.addWidget(self.outputFolderPath)
        outputFolderLayout.addWidget(self.outputFolderButton)

        # Name input
        self.nameLabel = QLabel("Name:")
        self.nameInput = QLineEdit(self)

        # Dropdown for Cambridge/Neuronexus
        self.dropdownLabel = QLabel("Select Option:")
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["Cambridge", "Neuronexus"])

        # Submit button
        self.submitButton = QPushButton("Submit")
        self.submitButton.clicked.connect(self.submit)

        # Add widgets to layout
        layout.addWidget(self.inputFolderLabel)
        layout.addLayout(inputFolderLayout)
        layout.addWidget(self.outputFolderLabel)
        layout.addLayout(outputFolderLayout)
        layout.addWidget(self.nameLabel)
        layout.addWidget(self.nameInput)
        layout.addWidget(self.dropdownLabel)
        layout.addWidget(self.dropdown)
        layout.addWidget(self.submitButton)

        self.setLayout(layout)
        self.setWindowTitle('Folder and Name Input')

    def browseInputFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
        if folder:
            self.inputFolderPath.setText(folder)

    def browseOutputFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder:
            self.outputFolderPath.setText(folder)

    def submit(self):
        self.inputFolder = self.inputFolderPath.text()
        self.outputFolder = self.outputFolderPath.text()
        self.name = self.nameInput.text()
        self.selectedOption = self.dropdown.currentText()
        self.close()

class InputDialog(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Layouts
        vbox = QVBoxLayout()

        # Input Folder
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel('Input Folder:')
        self.folder_line = QLineEdit()
        self.folder_button = QPushButton('Browse...')
        self.folder_button.clicked.connect(self.select_folder)

        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_line)
        folder_layout.addWidget(self.folder_button)

        # Input File
        file_layout = QHBoxLayout()
        self.file_label = QLabel('Input File:')
        self.file_line = QLineEdit()
        self.file_button = QPushButton('Browse...')
        self.file_button.clicked.connect(self.select_file)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_line)
        file_layout.addWidget(self.file_button)

        # Name
        name_layout = QHBoxLayout()
        self.name_label = QLabel('Name:')
        self.name_line = QLineEdit()

        name_layout.addWidget(self.name_label)
        name_layout.addWidget(self.name_line)

        # Submit Button
        self.submit_button = QPushButton('Submit')
        self.submit_button.clicked.connect(self.submit)

        # Add all layouts to the main layout
        vbox.addLayout(folder_layout)
        vbox.addLayout(file_layout)
        vbox.addLayout(name_layout)
        vbox.addWidget(self.submit_button)

        self.setLayout(vbox)
        self.setWindowTitle('Input Form')

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            self.folder_line.setText(folder)

    def select_file(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Select File')
        if file:
            self.file_line.setText(file)

    def submit(self):
        self.folder = self.folder_line.text()
        self.file = self.file_line.text()
        self.name = self.name_line.text()
