import vtk
from PyQt5.QtWidgets import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class Color:
    def __init__(self, r, g, b, alpha):
        self.r = r
        self.g = g
        self.b = b
        self.a = alpha

class GUIViewer(QVTKRenderWindowInteractor):
    def __init__(self, parent, engine):
        QVTKRenderWindowInteractor.__init__(self, parent)

        self.part_num = 0
        self.part_list = []
        self.point_list = []
        self.color_list = []
        self.glyph3D_list = []

        self.engine = engine
        self.resetCamera = True
        self.renderer = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.GetRenderWindow().GetInteractor()
        self.create_actor()

        self.renderer.SetBackground(255, 255, 255)
        camera = self.renderer.GetActiveCamera()
        camera.SetViewUp(0.5, 0, 0)
        camera.SetPosition(0.1245, 0.1139, 0.2932)
        self.renderer.SetActiveCamera(camera)

        transform = vtk.vtkTransform()
        transform.Translate(0.0, 0.0, 0.0)

        axes = vtk.vtkAxesActor()
        #  The axes are positioned with a user transform
        axes.SetUserTransform(transform)


    def create_voxel(self):
        numberOfVertices = 8

        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(1, 0, 0)
        points.InsertNextPoint(0, 1, 0)
        points.InsertNextPoint(1, 1, 0)
        points.InsertNextPoint(0, 0, 1)
        points.InsertNextPoint(1, 0, 1)
        points.InsertNextPoint(0, 1, 1)
        points.InsertNextPoint(1, 1, 1)

        voxel = vtk.vtkVoxel()
        for i in range(0, numberOfVertices):
            voxel.GetPointIds().SetId(i, i)

        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        ugrid.InsertNextCell(voxel.GetCellType(), voxel.GetPointIds())

        gfilter = vtk.vtkGeometryFilter()
        gfilter.SetInput(ugrid)
        gfilter.Update()
        return gfilter

    def create_actor(self):
        self.part_num = len(self.engine.model)
        for i in range(self.part_num):
            # Set up point_list and color_list
            self.point_list.append(vtk.vtkPoints())
            self.color_list.append(vtk.vtkUnsignedCharArray())
            self.color_list[i].SetName("colors")
            self.color_list[i].SetNumberOfComponents(4)

            # Create polydata by setting up points information and color information
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(self.point_list[i])
            polydata.GetPointData().SetScalars(self.color_list[i])

            # create cell
            voxel = self.create_voxel()

            # Set up Glyph3D data representation: color, data representation form, input data and et.
            self.glyph3D_list.append(vtk.vtkGlyph3D())
            self.glyph3D_list[i].SetColorModeToColorByScalar()
            self.glyph3D_list[i].SetSource(voxel.GetOutput())       # Set up data representation form
            self.glyph3D_list[i].SetInput(polydata)     # Set input data
            self.glyph3D_list[i].ScalingOff()
            self.glyph3D_list[i].Update()

            # mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInput(self.glyph3D_list[i].GetOutput())

            # actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty()
            self.renderer.AddActor(actor)
            self.part_list.append(actor)

        transform = vtk.vtkTransform()
        transform.Translate(0.0, 0.0, 0.0)
        axes = vtk.vtkAxesActor()
        #  The axes are positioned with a user transform
        axes.SetUserTransform(transform)

        # the actual text of the axis label can be changed:
        axes.SetXAxisLabelText("")
        axes.SetYAxisLabelText("")
        axes.SetZAxisLabelText("")

        # self.renderer.AddActor(axes)

    def set_bb_color(self, model, ind, n, x_zero_index, y_zero_index, z_zero_index, r, g, b):
        # alpha = 255
        alpha = 173
        for i in range(len(model[0])):
            self.point_list[ind].InsertNextPoint(0 - x_zero_index, 0 - y_zero_index, i - z_zero_index)
            self.color_list[ind].InsertTuple4(n, r, g, b, alpha)
            n = n + 1

            self.point_list[ind].InsertNextPoint(0 - x_zero_index, len(model[0]) - y_zero_index, i - z_zero_index)
            self.color_list[ind].InsertTuple4(n, r, g, b, alpha)
            n = n + 1

            self.point_list[ind].InsertNextPoint(len(model[0]) - x_zero_index, 0 - y_zero_index, i - z_zero_index)
            self.color_list[ind].InsertTuple4(n, r, g, b, alpha)
            n = n + 1

            self.point_list[ind].InsertNextPoint(len(model[0]) - x_zero_index, len(model[0]) - y_zero_index,
                                                 i - z_zero_index)
            self.color_list[ind].InsertTuple4(n,  r, g, b, alpha)
            n = n + 1

            self.point_list[ind].InsertNextPoint(0 - x_zero_index, i - y_zero_index, 0 - z_zero_index)
            self.color_list[ind].InsertTuple4(n,  r, g, b, alpha)
            n = n + 1

            self.point_list[ind].InsertNextPoint(0 - x_zero_index, i - y_zero_index, len(model[0]) - z_zero_index)
            self.color_list[ind].InsertTuple4(n,  r, g, b, alpha)
            n = n + 1

            self.point_list[ind].InsertNextPoint(len(model[0]) - x_zero_index, i - y_zero_index, 0 - z_zero_index)
            self.color_list[ind].InsertTuple4(n,  r, g, b, alpha)
            n = n + 1

            self.point_list[ind].InsertNextPoint(len(model[0]) - x_zero_index, i - y_zero_index,
                                                 len(model[0]) - z_zero_index)
            self.color_list[ind].InsertTuple4(n,  r, g, b, alpha)
            n = n + 1

            self.point_list[ind].InsertNextPoint(i - x_zero_index, 0 - y_zero_index, 0 - z_zero_index)
            self.color_list[ind].InsertTuple4(n, r, g, b, alpha)
            n = n + 1

            self.point_list[ind].InsertNextPoint(i - x_zero_index, 0 - y_zero_index, len(model[0]) - z_zero_index)
            self.color_list[ind].InsertTuple4(n,  r, g, b, 173)
            n = n + 1

            self.point_list[ind].InsertNextPoint(i - x_zero_index, len(model[0]) - y_zero_index, 0 - z_zero_index)
            self.color_list[ind].InsertTuple4(n,  r, g, b, 173)
            n = n + 1

            self.point_list[ind].InsertNextPoint(i - x_zero_index, len(model[0]) - y_zero_index,
                                                 len(model[0]) - z_zero_index)
            self.color_list[ind].InsertTuple4(n,  r, g, b, 173)
            n = n + 1

    def update_actor(self):
        models = self.engine.model

        colors = []
        colors.append(Color(241, 95, 45, 255))  # 0
        # colors.append(Color(246, 102, 39, 255)) # .........if(chair_2_legs),open this.......... #
        colors.append(Color(0, 202, 122, 255))  # 1
        colors.append(Color(0, 141, 237, 255))  # 2
        colors.append(Color(36, 160, 191, 255))  # 3
        colors.append(Color(255, 130, 58, 255))  # 4
        colors.append(Color(140, 255, 128, 255))  # 5

        for ind in range(len(models)):
            self.point_list[ind].Reset()
            self.color_list[ind].Reset()

            model = models[ind]['model']

            part_name = models[ind]['name']
            part_label = int(part_name[-1])

            translate_vector = models[ind]['translate_vector']
            dims_vector = models[ind]['dim_vecetor']

            x_zero_index = 0
            y_zero_index = 0
            z_zero_index = 0

            print len(model[0]), len(model[0]), len(model[0])
            n = 0
            for i in range(len(model[0])):
                for j in range(len(model[1])):
                    for k in range(len(model[2])):
                        c = model[i, j, k]
                        if c > 0.5:
                            if part_label == 0:
                                self.point_list[ind].InsertNextPoint(i - x_zero_index, j - y_zero_index,
                                                                     k - z_zero_index)
                                self.color_list[ind].InsertTuple4(n, 241, 95, 45, 200)
                            if part_label == 1:
                                self.point_list[ind].InsertNextPoint(i - x_zero_index, j - y_zero_index,
                                                                     k - z_zero_index)
                                self.color_list[ind].InsertTuple4(n, 0, 202, 122, 200)
                            if part_label == 2:
                                self.point_list[ind].InsertNextPoint(i - x_zero_index, j - y_zero_index,
                                                                     k - z_zero_index)
                                self.color_list[ind].InsertTuple4(n, 0, 141, 237, 200)
                            if part_label == 3:
                                self.point_list[ind].InsertNextPoint(i - x_zero_index, j - y_zero_index,
                                                                     k - z_zero_index)
                                self.color_list[ind].InsertTuple4(n, 36, 160, 191, 200)
                            if part_label == 4:
                                self.point_list[ind].InsertNextPoint(i - x_zero_index, j - y_zero_index,
                                                                     k - z_zero_index)
                                self.color_list[ind].InsertTuple4(n, 255, 130, 58, 200)
                            if part_label == 5:
                                self.point_list[ind].InsertNextPoint(i - x_zero_index, j - y_zero_index,
                                                                     k - z_zero_index)
                                self.color_list[ind].InsertTuple4(n, 140, 255, 128, 200)
                            if part_label == 6:
                                self.point_list[ind].InsertNextPoint(i - x_zero_index, j - y_zero_index,
                                                                     k - z_zero_index)
                                self.color_list[ind].InsertTuple4(n, 200, 240, 255, 200)
                            n = n + 1
                        pass

            self.set_bb_color(model, ind, n, x_zero_index, y_zero_index, z_zero_index, colors[part_label].r, colors[part_label].g, colors[part_label].b)

            trans = vtk.vtkTransform()
            transformMatrix = vtk.vtkMatrix4x4()
            for i in range(0, 4):
                for j in range(0, 4):
                    if i == j and (i == 1 or i == 2 or i == 0):
                        transformMatrix.SetElement(i, j, float(dims_vector[i]) / 32.0)
                    elif j == 3 and i <= 2:
                        transformMatrix.SetElement(i, j, float(translate_vector[i]) - float(dims_vector[i]) / 2)
                    elif i == 3 and j == 3:
                        transformMatrix.SetElement(i, j, 1)
                    elif i == 3 and j <= 2:
                        transformMatrix.SetElement(i, j, 0)
            trans.SetMatrix(transformMatrix)
            self.part_list[ind].SetUserTransform(trans)

            print translate_vector[0], translate_vector[1], translate_vector[2]

            print "part index: %r center: %r" % (ind, self.part_list[ind].GetCenter())
            print "part index: %r position: %r" % (ind, self.part_list[ind].GetPosition())
            print "part index: %r xRange: %r" % (ind, self.part_list[ind].GetXRange())
            print "part index: %r length: %r" % (ind, self.part_list[ind].GetLength())
            print "part index: %r translate_vector: %r" % (ind, translate_vector)
            print "part index: %r dims_vector: %r" % (ind, dims_vector)

        # self.glyph3D.Modified()
        for i in range(len(self.glyph3D_list)):
            self.glyph3D_list[i].Modified()

        if self.resetCamera:
            self.renderer.ResetCamera()

            self.resetCamera = False
        self.update()
        QApplication.processEvents()

        for i in range(len(self.part_list)):
            print self.part_list[i].GetPosition()


    def save_image2(self):
        file_path = QFileDialog.getSaveFileName(self, "saveFlle", str(self.engine.name) + ".png",
                                                filter="png (*.png *.)")
        current_file_path = file_path[0]
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.GetRenderWindow())
        window_to_image_filter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(unicode(current_file_path))
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    def save_image1(self, file_path):
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.GetRenderWindow())
        window_to_image_filter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(unicode(file_path) + '/image_' + str(self.engine.index) + '.png')
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    def set_camera(self, x, y, z):
        camera = self.renderer.GetActiveCamera()
        camera.SetViewUp(x, y, z)
        self.renderer.SetActiveCamera(camera)