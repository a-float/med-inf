from vtk import (
    vtkImageShiftScale,
    vtkDICOMImageReader,
    vtkAlgorithmOutput,
    vtkContourFilter,
    vtkActor,
    vtkPolyDataMapper,
    vtkOutlineFilter,
    vtkNamedColors,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkSliderRepresentation2D,
    vtkSliderWidget,
    vtkInteractorStyleTrackballCamera,
    vtkVolumeProperty,
    vtkColorTransferFunction,
    vtkVolume,
    vtkLookupTable,
    vtkSmartVolumeMapper,
    vtkPiecewiseFunction,
)

DICOM_DIR = "mr_brainixA"
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


class OpacityCallback:
    def __init__(
        self,
        x: int,
        y: float,
        composite_opacity: vtkPiecewiseFunction,
        color,
        color_function,
    ):
        self.x = x
        self.y = y
        self.composite_opacity = composite_opacity
        self.composite_opacity.AddPoint(x, y)
        self.color = color
        self.color_function = color_function
        self.color_function.AddRGBPoint(x, *color)

    def __call__(self, caller, ev):
        slider_value = caller.GetSliderRepresentation().GetValue()
        print(slider_value)
        self.composite_opacity.RemovePoint(self.x)
        self.color_function.RemovePoint(self.x)
        self.composite_opacity.AddPoint(slider_value, self.y)
        self.color_function.AddRGBPoint(slider_value, *self.color)
        self.x = slider_value


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


def build_slider_rep(
    x: float, value: float, label: str, image_range
) -> vtkSliderRepresentation2D:
    slider_rep = vtkSliderRepresentation2D()
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(x - 0.15, 0.05)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(x + 0.15, 0.05)
    slider_rep.SetMinimumValue(image_range[0])
    slider_rep.SetMaximumValue(image_range[1])
    slider_rep.SetEndCapWidth(0.03)
    slider_rep.SetTitleHeight(0.02)
    slider_rep.SetValue(value)
    slider_rep.SetTitleText(label)

    return slider_rep


def connect_outline_actor(outline_mapper: vtkPolyDataMapper) -> vtkActor:
    outline_actor = vtkActor()
    outline_actor.SetMapper(outline_mapper)
    outline_actor.GetProperty().SetLineWidth(3)
    outline_actor.GetProperty().SetColor(colors.GetColor3d("Gray"))

    return outline_actor


def build_volume_actor(
    mapper: vtkSmartVolumeMapper,
    composite_opacity: vtkPiecewiseFunction,
    color: vtkColorTransferFunction,
) -> vtkVolume:
    volume_property = vtkVolumeProperty()
    volume_property.ShadeOff()
    volume_property.SetInterpolationTypeToLinear()
    volume_property.SetScalarOpacity(composite_opacity)  # composite first.

    volume_property.SetColor(color)

    actor = vtkVolume()
    actor.SetMapper(mapper)
    actor.SetProperty(volume_property)
    return actor


if __name__ == "__main__":
    colors = vtkNamedColors()
    reader = build_reader()
    image_data = reader.GetOutput()
    image_range = image_data.GetScalarRange()
    n_frames = image_data.GetDimensions()[2]

    window_filter = connect_window_filter(input_connection=reader.GetOutputPort())
    contour_filter = connect_contour_filter(input_data=image_data)
    contour_mapper = connect_contour_mapper(
        input_connection=contour_filter.GetOutputPort()
    )
    contour_mapper.SetScalarRange(image_data.GetScalarRange())

    contour_actor = build_contour_actor(contour_mapper=contour_mapper)

    composite_opacity = vtkPiecewiseFunction()
    color_function = vtkColorTransferFunction()

    volume_mapper = vtkSmartVolumeMapper()
    volume_mapper.SetBlendModeToComposite()
    volume_mapper.SetInputData(image_data)

    volume_actor = build_volume_actor(
        mapper=volume_mapper, composite_opacity=composite_opacity, color=color_function
    )
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

    # renderer.AddActor(contour_actor)
    # renderer.AddActor(outline_actor)
    renderer.AddActor(volume_actor)

    xs = [40, 80, 255]
    ys = [0, 0.2, 0]
    colors = [[0, 0, 1], [1, 0, 0], [1, 1, 1]]
    slider_reps = [
        build_slider_rep(x=0.17, value=xs[0], label="start", image_range=image_range),
        build_slider_rep(x=0.5, value=xs[1], label="mid", image_range=image_range),
        build_slider_rep(x=0.83, value=xs[2], label="end", image_range=image_range),
    ]
    sliders = [vtkSliderWidget() for _ in range(3)]
    for i, slider in enumerate(sliders):
        print(i, slider)
        slider.SetInteractor(interactor)
        slider.SetRepresentation(slider_reps[i])
        slider.SetAnimationModeToAnimate()
        slider.EnabledOn()
        slider.AddObserver(
            "EndInteractionEvent",
            OpacityCallback(
                x=xs[i],
                y=ys[i],
                composite_opacity=composite_opacity,
                color=colors[i],
                color_function=color_function,
            ),
        )

    render_window.SetSize(500, 500)
    render_window.Render()

    style = vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    interactor.Initialize()
    interactor.Start()
