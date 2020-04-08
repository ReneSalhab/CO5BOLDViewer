from  traits.api import HasTraits
from  mayavi import mlab
from qtpy import  QtWidgets

import numpy as np

class MayaViWidget(QtWidgets.QWidget):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plotter = ThreeDPlotWidget()

        self.ui = self.plotter.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)

    def plot(self, data):
        self.plotter.plot(data)

class ThreeDPlotWidget(HasTraits):

    def __init__(self):
        HasTraits.__init__(self)

    def plot(self, data):
        mlab.pipeline.volume(mlab.pipeline.scalar_field(data))
