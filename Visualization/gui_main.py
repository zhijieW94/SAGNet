import os
import scipy.io as sio

from gui_viewer import *
from ConstrainedOpt import ConstrainedOpt

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
import qdarkstyle

class MySubWindow(QMdiSubWindow):
    signal_save_image = pyqtSignal(str)

class MainWindow(QMainWindow):
    signal_save_images = pyqtSignal(str)
    signal_setCamera = pyqtSignal(float, float, float)

    def __init__(self, width, height, parent=None):
        QMainWindow.__init__(self, parent)
        self.width = width
        self.height = height
        self.resize(width, height)
        self.mdi_Area = QMdiArea()
        self.mdi_Area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdi_Area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.camerax = 0
        self.cameray = 0
        self.cameraz = 0

        self.files = list()
        self.models = list()
        self.models_count = 0

        self.setCentralWidget(self.mdi_Area)

        self.mdi_Win = list()
        self.frame = list()
        self.viewerWidget = list()

        self.banch = 8
        self.current_index = -1

        self.file_ind = 0

        # create openAction
        openAction = QAction("&Open", self)
        openAction.setShortcut(QKeySequence.Open)
        openAction.setToolTip("Open files")
        openAction.setStatusTip("Open files")
        openAction.triggered.connect(self.open_file)

        openoneAction = QAction("&Openone", self)
        openoneAction.setShortcut(QKeySequence.Open)
        openoneAction.setToolTip("Open a file")
        openoneAction.setStatusTip("Open a file")
        openoneAction.triggered.connect(self.open_one_file)

        # preAction = QAction("&PreGroup", self)
        # preAction.setShortcut(QKeySequence.Open)
        # preAction.setToolTip("Show pre group model")
        # preAction.setStatusTip("Show pre group model")
        # preAction.triggered.connect(self.pre_group)
        # # self.connect(preAction, SIGNAL("triggered()"), self.pre_group)

        nextAction = QAction("&NextGroup", self)
        nextAction.setShortcut(QKeySequence.Open)
        nextAction.setToolTip("Show next group model")
        nextAction.setStatusTip("Show next group model")
        nextAction.triggered.connect(self.next_group)

        saveAction = QAction("&SaveAll", self)
        saveAction.setShortcut(QKeySequence.Open)
        saveAction.setToolTip("Save all models as Image")
        saveAction.setStatusTip("Save all models as Image")
        saveAction.triggered.connect(self.save_group)

        saveOneAction = QAction("&SaveCurrent", self)
        saveOneAction.setShortcut(QKeySequence.Open)
        saveOneAction.setToolTip("Save current model")
        saveOneAction.setStatusTip("Save current model")
        saveOneAction.triggered.connect(self.save_one)

        # create toolbar
        toolbar = self.addToolBar("tool")
        toolbar.setMovable(False)
        toolbar.setObjectName("ToolBar")
        toolbar.addAction(openAction)
        toolbar.addAction(openoneAction)
        toolbar.addAction(nextAction)
        toolbar.addAction(saveAction)
        toolbar.addAction(saveOneAction)

        # toolbar.addAction(preAction)

    def closeEvent(self, event):
        for i in range(len(self.mdi_Win)):
            self.mdi_Win[i].opt_engine.quit()

    def open_all_file_for_model(self, file_name):
        mat_list = []
        dir_name = ""
        for i in range (5):
            file_path = dir_name + file_name + "_" + str(i) + ".mat"
            data = sio.loadmat(file_path)
            mat_list.append(data)


    def get_object_model(self, dir_path):
        mat_file_list = []
        transform_file_list = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.mat'):
                mat_file_list.append(file_name)
            elif file_name.endswith('.txt'):
                transform_file_list.append(file_name)

        if len(mat_file_list) != len(transform_file_list):
            raise KeyError("The number of mat files don't match the number of transform files")

        label_list = []
        for mat_file in mat_file_list:
            cur_label = mat_file[-5: -4]
            label_list.append(cur_label)

        if len(mat_file_list[0]) == 5 and mat_file_list[0][0].isdigit():
            testing_samples = True
        else:
            testing_samples = False

        models = []
        cur_file_name = mat_file_list[0][0:-6]
        for cur_label in label_list:
            c_label = int(cur_label)

            if testing_samples:
                current_mat_file_name = "%d.mat" % (c_label)
                current_transform_file_name = "%d_transform_info.txt" % (c_label)
            else:
                current_transform_file_name = "%s_%d_transform_info.txt" % (cur_file_name, c_label)
                current_mat_file_name = "%s_%d.mat" % (cur_file_name, c_label)

            part_name = "%s_%d" % (cur_file_name, c_label)

            current_mat_file_path = "%s/%s" % (dir_path, current_mat_file_name)
            current_transform_file_path = "%s/%s" % (dir_path, current_transform_file_name)

            file = open(current_transform_file_path, 'r')

            is_bbox_vector = False
            is_min_max_value = False

            file_lines = file.readlines()
            for line in file_lines:
                if line == "\r\n":
                    is_bbox_vector = False
                    is_min_max_value = False
                    continue
                if line.find("vector for representation") != -1:
                    is_min_max_value = True
                    is_bbox_vector = False
                    continue
                if line.find("Min_Max Value") != -1:
                    is_bbox_vector = False
                    is_min_max_value = True
                    continue
                if is_bbox_vector:
                    value_list = line.split(' ')
                if is_min_max_value:
                    line = line.strip()
                    line = line.strip(' \t\n\r')
                    value_list = line.split(' ')
                    break

            translate_vector = [float(value_list[0]), float(value_list[1]), float(value_list[2])]

            value_list[3] = abs(float(value_list[3]))
            value_list[4] = abs(float(value_list[4]))
            value_list[5] = abs(float(value_list[5]))

            min_max_value_list = [(float(value_list[0]) - float(value_list[3]) / 2),
                                  (float(value_list[1]) - float(value_list[4]) / 2),
                                  (float(value_list[2]) - float(value_list[5]) / 2),
                                  (float(value_list[0]) + float(value_list[3]) / 2),
                                  (float(value_list[1]) + float(value_list[4]) / 2),
                                  (float(value_list[2]) + float(value_list[5]) / 2)]

            # These three values represent the length, width, height of a bounding box
            dim_list = [float(value_list[3]), float(value_list[4]), float(value_list[5])]

            mat_data = sio.loadmat(current_mat_file_path)
            array = mat_data['voxels3D']
            model = {'name': part_name, 'model': array, 'translate_vector': translate_vector,
                     'dim_vecetor': dim_list, 'min_max_value': min_max_value_list}
            models.append(model)

        model_dict = {'name': cur_file_name, 'model': models}
        # self.models.append(model_dict)

        return model_dict

    def open_one_file(self):
        self.models_count = 0
        dir_ = QFileDialog.getExistingDirectory(None, 'Select a folder:', '.', QFileDialog.ShowDirsOnly)
        self.files.append(dir_)

        model_dict = self.get_object_model(dir_)
        self.models.append(model_dict)
        self.models_count = 1
        self.current_index = self.current_index + 1
        self.banch = 1
        self.view_model()

    def open_file(self):
        self.banch = 8
        self.models_count = 0
        dir_ = QFileDialog.getExistingDirectory(None, 'Select a folder:', '.', QFileDialog.ShowDirsOnly)
        for filename in os.listdir(dir_):
            self.files.append(dir_ + '/' + filename)
        obj_names = [f_path for f_path in os.listdir(dir_) if os.path.isdir(os.path.join(dir_, f_path))]
        obj_names.sort()

        # fileList contains all the file pathes beyond a certain category
        obj_file_pathes = [os.path.join(dir_, f_path) for f_path in obj_names if os.path.isdir(os.path.join(dir_, f_path))]
        self.files = obj_file_pathes

        if obj_names[0].find('iter_') != -1:
            def key_word(item):
                return int(item.split('_')[-1])

            self.files.sort(key=key_word)

        for idx in range(0, self.banch):
            if (len(self.files) > 0):
                obj_file_path = self.files.pop()
                obj_file_path = obj_file_path.strip(' \t\n\r')
                print 'current path: ' + obj_file_path
                model_dict = self.get_object_model(obj_file_path)
                self.models.append(model_dict)
                self.models_count = self.models_count + 1

        self.current_index = 0
        self.banch = self.models_count
        self.view_model()

    # TODO This function is important
    def view_model(self):
        start = self.current_index
        banch = self.banch
        end = start + banch

        width = (self.width * 2 / 8) * 0.95
        height = (self.height / 2) * 0.95
        mainWidth = width + 10
        mainHeight = height + 10
        self.setWindowTitle("model_view      models_count:" + str(len(self.files)))

        for index in range(start, end):
            if (len(self.models) <= 0):
                break
            # Obtain one new model to display
            model = self.models.pop()
            # Update the number of models to show
            self.models_count = self.models_count - 1
            # Add one more sub-frame to the main window
            self.frame.append(QFrame())
            self.mdi_Win.append(MySubWindow())
            self.mdi_Win[index].opt_engine = ConstrainedOpt(model, index)
            self.mdi_Win[index].setWindowTitle("model_" + model['name'])
            self.mdi_Win[index].setGeometry(0, 0, mainWidth, mainHeight)

            self.viewerWidget.append(
                GUIViewer(self.frame[index], self.mdi_Win[index].opt_engine))
            self.viewerWidget[index].resize(width, height)

            viewerBox = QVBoxLayout()
            viewerBox.addWidget(self.viewerWidget[index])
            self.frame[index].setLayout(viewerBox)
            self.mdi_Win[index].setWidget(self.frame[index])

            self.viewerWidget[index].interactor.Initialize()

            self.mdi_Win[index].opt_engine.signal_update_voxels.connect(self.viewerWidget[index].update_actor)
            self.mdi_Win[index].signal_save_image.connect(self.viewerWidget[index].save_image2)
            self.signal_save_images.connect(self.viewerWidget[index].save_image1)

            self.mdi_Win[index].opt_engine.start()
            self.mdi_Area.addSubWindow(self.mdi_Win[index])
            self.mdi_Win[index].show()
            print 'success'

    # def pre_group(self):
    #     self.current_index = self.current_index - self.banch
    #     if self.current_index < 0:
    #         self.current_index = 0
    #     for i in range(len(self.mdi_Win)):
    #         self.mdi_Win[i].close()
    #     self.mdi_Win[:] = []
    #     self.frame[:] = []
    #     self.viewerWidget[:] = []
    #     self.view_model()
    #     # self.models_count = 0

    def next_group(self):
        self.banch = 8
        for i in range(len(self.mdi_Win)):
            self.mdi_Win[i].close()
        self.mdi_Win[:] = []
        self.frame[:] = []
        self.viewerWidget[:] = []
        self.models[:] = []

        for idx in range(0, self.banch):
            if (len(self.files) > 0):
                file_path = self.files.pop()
                print 'current path: ' + file_path
                file_path = file_path.strip(' \t\n\r')
                model_dict = self.get_object_model(file_path)
                self.models.append(model_dict)
                self.models_count = 1 + self.models_count

        self.current_index = 0
        self.banch = self.models_count
        self.view_model()

    def save_group(self):
        file_path = QFileDialog.getExistingDirectory(self, "Open a folder", ".", QFileDialog.ShowDirsOnly)
        self.signal_save_images.emit(file_path)

    def save_one(self):
        self.mdi_Area.currentSubWindow().signal_save_image.emit('save_image')

    def setCamera(self):
        self.camerax = float(unicode(self.valuex.text()))
        self.cameray = float(unicode(self.valuey.text()))
        self.cameraz = float(unicode(self.valuez.text()))
        valuex = self.camerax
        valuey = self.cameray
        valuez = self.cameraz
        self.signal_setCamera.emit(valuex, valuey, valuez)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    desktopWidget = QApplication.desktop()
    screenRect = desktopWidget.screenGeometry()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainWindow(screenRect.width(),screenRect.height())
    window.setWindowTitle("model_view")
    window.show()
    sys.exit(app.exec_())
