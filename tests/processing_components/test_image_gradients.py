"""Unit tests for testing support

"""

import logging
import unittest

from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.processing_components.image.gradients import image_gradients
from rascil.processing_components.image.operations import export_image_to_fits, show_image, import_image_from_fits

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)

class TestPrimaryBeams(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path, rascil_data_path
        self.dir = rascil_path('test_results')

        self.show = False
        self.persist = False
    
    def test_create_gradient(self):
        real_vp = import_image_from_fits(rascil_data_path('models/MID_GRASP_VP_real.fits'))
        gradx, grady = image_gradients(real_vp)
        
        gradxx, gradxy = image_gradients(gradx)
        gradyx, gradyy = image_gradients(grady)

        gradx.data *= real_vp.data
        grady.data *= real_vp.data
        gradxx.data *= real_vp.data
        gradxy.data *= real_vp.data
        gradyx.data *= real_vp.data
        gradyy.data *= real_vp.data

        if self.show:
            import matplotlib.pyplot as plt
            plt.clf()
            show_image(gradx, title='gradx')
            plt.show(block=False)
            plt.clf()
            show_image(grady, title='grady')
            plt.show(block=False)
        if self.persist:
            export_image_to_fits(gradx, "%s/test_image_gradients_gradx.fits" % (self.dir))
            export_image_to_fits(grady, "%s/test_image_gradients_grady.fits" % (self.dir))

        if self.show:
            import matplotlib.pyplot as plt
            plt.clf()
            show_image(gradxx, title='gradxx')
            plt.show(block=False)
            plt.clf()
            show_image(gradxy, title='gradxy')
            plt.show(block=False)
            plt.clf()
            show_image(gradyx, title='gradyx')
            plt.show(block=False)
            plt.clf()
            show_image(gradyy, title='gradyy')
            plt.show(block=False)
        if self.persist:
            export_image_to_fits(gradxx, "%s/test_image_gradients_gradxx.fits" % (self.dir))
            export_image_to_fits(gradxy, "%s/test_image_gradients_gradxy.fits" % (self.dir))
            export_image_to_fits(gradyx, "%s/test_image_gradients_gradyx.fits" % (self.dir))
            export_image_to_fits(gradyy, "%s/test_image_gradients_gradyy.fits" % (self.dir))


if __name__ == '__main__':
    unittest.main()
