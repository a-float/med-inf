from vtk import (vtkImageShiftScale, vtkDICOMImageReader, vtkAlgorithmOutput, vtkContourFilter, vtkActor,
                 vtkPolyDataMapper, vtkOutlineFilter, vtkNamedColors, vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor,
                 vtkSliderRepresentation2D, vtkSliderWidget, vtkInteractorStyleTrackballCamera, vtkVolumeProperty, vtkColorTransferFunction,
                 vtkVolume, vtkLookupTable)
DICOM_DIR = 'mr_brainixA'
WINDOW_WIDTH = 750
WINDOW_CENTER = 100
CONTOURS = 1


class ContourCallback:
    def __init__(self, contour_filter: vtkContourFilter):
        self.contour_filter = contour_filter

    def __call__(self, caller, ev):
        slider_value = caller.GetSliderRepresentation().GetValue()
        print(slider_value)
        self.contour_filter.SetValue(0, slider_value)


# NOT used for now, maybe will be useful later
def get_diverging_lut():
    """
    See: [Diverging Color Maps for Scientific Visualization](https://www.kennethmoreland.com/color-maps/)
                       start point         midPoint            end point
     cool to warm:     0.230, 0.299, 0.754 0.865, 0.865, 0.865 0.706, 0.016, 0.150
     purple to orange: 0.436, 0.308, 0.631 0.865, 0.865, 0.865 0.759, 0.334, 0.046
     green to purple:  0.085, 0.532, 0.201 0.865, 0.865, 0.865 0.436, 0.308, 0.631
     blue to brown:    0.217, 0.525, 0.910 0.865, 0.865, 0.865 0.677, 0.492, 0.093
     green to red:     0.085, 0.532, 0.201 0.865, 0.865, 0.865 0.758, 0.214, 0.233

    :return:
    """
    ctf = vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    # Cool to warm.
    ctf.AddRGBPoint(0.0, 0.230, 0.299, 0.754)
    ctf.AddRGBPoint(0.5, 0.865, 0.865, 0.865)
    ctf.AddRGBPoint(1.0, 0.706, 0.016, 0.150)

    table_size = 256
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(table_size)
    lut.Build()

    for i in range(0, table_size):
        rgba = list(ctf.GetColor(float(i) / table_size))
        rgba.append(1)
        lut.SetTableValue(i, rgba)

    return lut


def build_reader() -> vtkDICOMImageReader:
    reader = vtkDICOMImageReader()
    reader.SetDirectoryName(DICOM_DIR)
    reader.Update()
    return reader


def connect_window_filter(input_connection: vtkAlgorithmOutput) -> vtkImageShiftScale:
    shiftScaleFilter = vtkImageShiftScale()
    shiftScaleFilter.SetOutputScalarTypeToUnsignedChar()
    shiftScaleFilter.SetInputConnection(input_connection)
    shiftScaleFilter.SetShift(-WINDOW_CENTER + 0.5 * WINDOW_WIDTH)
    shiftScaleFilter.SetScale(255.0 / WINDOW_WIDTH)
    shiftScaleFilter.SetClampOverflow(True)


def connect_contour_filter(input_data) -> vtkContourFilter:
    contour_filter = vtkContourFilter()
    contour_filter.SetInputData(input_data)
    contour_filter.GenerateValues(CONTOURS, 10, 10)
    return contour_filter


def connect_contour_mapper(input_connection: vtkAlgorithmOutput) -> vtkPolyDataMapper:
    contour_mapper = vtkPolyDataMapper()
    contour_mapper.SetInputConnection(input_connection)
    # lut = get_diverging_lut()
    # contour_mapper.SetLookupTable(lut)
    return contour_mapper


def build_contour_actor(contour_mapper: vtkPolyDataMapper) -> vtkActor:
    contour_actor = vtkActor()
    contour_actor.SetMapper(contour_mapper)
    contour_actor.GetProperty().SetLineWidth(3)
    return contour_actor


def build_slider_rep() -> vtkSliderRepresentation2D:
    slider_rep = vtkSliderRepresentation2D()
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(.7, .1)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(.9, .1)
    slider_rep.SetMinimumValue(0)
    slider_rep.SetMaximumValue(500)
    slider_rep.SetValue(10)
    slider_rep.SetTitleText("iso")

    return slider_rep


def connect_outline_actor(outline_mapper: vtkPolyDataMapper) -> vtkActor:
    outline_actor = vtkActor()
    outline_actor.SetMapper(outline_mapper)
    outline_actor.GetProperty().SetLineWidth(3)
    outline_actor.GetProperty().SetColor(colors.GetColor3d("Gray"))

    return outline_actor


if __name__ == '__main__':
    colors = vtkNamedColors()
    reader = build_reader()
    image_data = reader.GetOutput()
    image_range = image_data.GetScalarRange()
    n_frames = image_data.GetDimensions()[2]

    window_filter = connect_window_filter(
        input_connection=reader.GetOutputPort())
    contour_filter = connect_contour_filter(input_data=image_data)
    contour_mapper = connect_contour_mapper(
        input_connection=contour_filter.GetOutputPort())
    contour_mapper.SetScalarRange(image_data.GetScalarRange())

    contour_actor = build_contour_actor(contour_mapper=contour_mapper)

    outline_filter = vtkOutlineFilter()
    outline_filter.SetInputData(image_data)

    outline_mapper = vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outline_filter.GetOutputPort())

    outline_actor = connect_outline_actor(outline_mapper=outline_mapper)

    renderer = vtkRenderer()
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName("IsoContours")

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    renderer.AddActor(contour_actor)
    renderer.AddActor(outline_actor)

    slider_rep = build_slider_rep()

    slider = vtkSliderWidget()
    slider.SetInteractor(interactor)
    slider.SetRepresentation(slider_rep)
    slider.SetAnimationModeToAnimate()
    slider.EnabledOn()
    slider.AddObserver('EndInteractionEvent', ContourCallback(
        contour_filter=contour_filter))

    # maybe needed later
    # volume_color = vtkColorTransferFunction()
    # volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    # volume_color.AddRGBPoint(500, 1.0, 0, 0)
    # volume_color.AddRGBPoint(1000, 1.0, 0, 0)
    # volume_property = vtkVolumeProperty()
    # volume_property.SetColor(volume_color)

    # volume = vtkVolume()
    # volume.SetProperty(volume_property)
    # renderer.AddVolume(volume)

    render_window.SetSize(500, 500)
    render_window.Render()

    style = vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    interactor.Initialize()
    interactor.Start()
