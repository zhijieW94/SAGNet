from PyQt5.QtCore import *

class ConstrainedOpt(QThread):
    signal_update_voxels = pyqtSignal(str)

    def __init__(self, model,index):
        QThread.__init__(self)
        self.model = model['model']
        # self.model = model
        self.name = model['name']

        self.index = index

    def run(self):
#        while True:
            self.update_voxel_model()

    def update_voxel_model(self):
        self.signal_update_voxels.emit('update_voxels')