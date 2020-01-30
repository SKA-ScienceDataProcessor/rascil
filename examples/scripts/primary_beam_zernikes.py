""" Make example primary beams by adding in random zernike terms
"""

import logging

import matplotlib.pyplot as plt
import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image import export_image_to_fits, qa_image, copy_image
from rascil.processing_components.imaging import create_image_from_visibility
from rascil.processing_components.imaging import create_vp_generic_numeric
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility import create_visibility

log = logging.getLogger(__name__)

if __name__ == '__main__':

    dir = '.'

    # Set up a short observation with MID
    dec = -45.0
    rmax = 1e3
    freq = 1.4e9
    frequency = numpy.linspace(freq, 1.5 * freq, 3)
    channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
    flux = numpy.array([[100.0], [100.0], [100.0]])
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs',
                           equinox='J2000')
    config = create_named_configuration('MIDR5', rmax=rmax)
    times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
    nants = config.xyz.shape[0]
    assert nants > 1
    assert len(config.names) == nants
    assert len(config.mount) == nants

    vis = create_visibility(config, times, frequency,
                            channel_bandwidth=channel_bandwidth,
                            phasecentre=phasecentre, weight=1.0,
                            polarisation_frame=PolarisationFrame('stokesI'))

    cellsize = 8 * numpy.pi / 180.0 / 280
    model = create_image_from_visibility(vis, npixel=512, cellsize=cellsize,
                                         override_cellsize=False)

    # These are the nolls that maintain left-right symmetry
    plt.clf()
    fig, axs = plt.subplots(4, 4, gridspec_kw={'hspace': 0, 'wspace': 0})
    ntrials = 16
    zernikes = list()
    default_vp = create_vp_generic_numeric(model, pointingcentre=None, diameter=15.0,
                                           blockage=0.0,
                                           taper='gaussian',
                                           edge=0.03162278, padding=2, use_local=True)

    key_nolls = [3, 5, 6, 7]
    for noll in key_nolls:
        zernike = {'coeff': 1.0, 'noll': noll}
        zernike['vp'] = create_vp_generic_numeric(model, pointingcentre=None,
                                                  diameter=15.0,
                                                  blockage=0.0,
                                                  taper='gaussian',
                                                  edge=0.03162278, zernikes=[zernike],
                                                  padding=2, use_local=True)
        zernikes.append(zernike)

    for trial in range(ntrials):
        coeffs = numpy.random.normal(0.0, 0.03, len(key_nolls))
        vp = copy_image(default_vp)
        for i in range(len(key_nolls)):
            vp.data += coeffs[i] * zernikes[i]['vp'].data

        vp.data = vp.data / numpy.max(numpy.abs(vp.data))
        vp_data = vp.data / numpy.max(numpy.abs(vp.data))
        vp.data = numpy.real(vp_data)
        print(trial, qa_image(vp))

        export = False
        if export:
            export_image_to_fits(vp,
                                 "%s/test_voltage_pattern_real_%s_trial%d.fits"
                                 % (dir, 'MID_RANDOM_ZERNIKES', trial))

        row = (trial - 1) // 4
        col = (trial - 1) - 4 * row
        ax = axs[row, col]
        ax.imshow(vp.data[0, 0], vmax=0.01, vmin=-0.001)
        # ax.set_title('Noll %d' % noll)
        ax.axis('off')

    plt.savefig("random_zernikes.png")
    plt.show()
