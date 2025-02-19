from spike_functs import *
import sys
from PyQt5.QtWidgets import QApplication
from bin_gui import FolderInput

def main():
    app = QApplication(sys.argv)
    folderInput = FolderInput()
    folderInput.show()
    app.exec_()

    # Retrieve the values from the GUI
    input_folder = folderInput.inputFolder
    output_folder = folderInput.outputFolder
    name = folderInput.name
    sel_probe = folderInput.selectedOption

    # Run the spike sorting process
    run_spike_sorting(input_folder, output_folder, name, sel_probe)

    print("All done! Results save to ", output_folder + '/results-' + name)

if __name__ == '__main__':
    main()
