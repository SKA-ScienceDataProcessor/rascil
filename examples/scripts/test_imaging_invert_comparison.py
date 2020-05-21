#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
import logging
import sys
import unittest
import time
import wtowers.wtowers as wtowers

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 10

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.operations import qa_image, show_image, export_image_to_fits,     smooth_image, copy_image
from rascil.processing_components.imaging.base import predict_skycomponent_visibility
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility,     create_unittest_model, create_unittest_components
from rascil.processing_components.skycomponent.operations import find_skycomponents, find_nearest_skycomponent,     insert_skycomponent
from rascil.processing_components.visibility.coalesce import convert_blockvisibility_to_visibility
from rascil.processing_components.imaging.base import invert_2d, predict_2d, shift_vis_to_image, normalize_sumwt
from rascil.processing_components.visibility.base import copy_visibility
from rascil.processing_components.imaging.ng import predict_ng, invert_ng
from rascil.processing_components.griddata.kernels import create_awterm_convolutionfunction


# In[3]:


rdir = './'
verbosity = True
dopol = False
dospectral = True
zerow = False
block = True
persist = True
npixel = 1024
low = create_named_configuration('LOWBD2', rmax=750.0)
freqwin = 21
blockvis = list()
ntimes = 5
times = numpy.linspace(-3.0, +3.0, ntimes) * numpy.pi / 12.0
        
if freqwin > 1:
    frequency = numpy.linspace(0.99e8, 1.01e8, freqwin)
    channelwidth = numpy.array(freqwin * [frequency[1] - frequency[0]])
else:
    frequency = numpy.array([1e8])
    channelwidth = numpy.array([1e6])
        
if dopol:
    blockvis_pol = PolarisationFrame('linear')
    image_pol = PolarisationFrame('stokesIQUV')
    f = numpy.array([100.0, 20.0, -10.0, 1.0])
else:
    blockvis_pol = PolarisationFrame('stokesI')
    image_pol = PolarisationFrame('stokesI')
    f = numpy.array([100.0])
        
if dospectral:
    flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in frequency])
else:
    flux = numpy.array([f])
        
phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
blockvis = ingest_unittest_visibility(low,
                                                   frequency,
                                                   channelwidth,
                                                   times,
                                                   blockvis_pol,
                                                   phasecentre,
                                                   block=block,
                                                   zerow=zerow)
        
vis = convert_blockvisibility_to_visibility(blockvis)

model = create_unittest_model(vis, image_pol, npixel=npixel, nchan=freqwin)
        
components = create_unittest_components(model, flux)
model = insert_skycomponent(model, components)
        
blockvis = predict_skycomponent_visibility(blockvis, components)
#blockvis = dft_skycomponent_visibility(blockvis, components)

blockvis1 = copy_visibility(blockvis)
vis1 = convert_blockvisibility_to_visibility(blockvis1)

# Calculate the model convolved with a Gaussian.
        
cmodel = smooth_image(model)
if persist: export_image_to_fits(model, '%s/test_imaging_2d_model.fits' % rdir)
if persist: export_image_to_fits(cmodel, '%s/test_imaging_2d_cmodel.fits' % rdir)


# In[4]:


print(qa_image(model))


# In[5]:


plt.rcParams['figure.figsize'] = 10, 10
show_image(cmodel)
plt.savefig("cmodel.png")


# In[6]:


# Find nw based on w_min, w_max
w_min = numpy.amin(vis.data['uvw'][:,2])
w_max = numpy.amax(vis.data['uvw'][:,2])

w_range = 2*numpy.amax((numpy.abs(w_min), numpy.abs(w_max)))
wstep = 3.0
nw = numpy.floor(w_range/wstep)
nw = int(1.1*nw)
if nw%2 == 0:
    nw = nw+1
print(w_min, w_max,w_range, wstep, nw)    
    


# In[7]:


#%timeit dirty_ng,_ = invert_ng(blockvis, model, normalize=True)


# In[10]:


# Make Rascil kernel
start = time.time()
gcfcf_2d = create_awterm_convolutionfunction(model, make_pb=None, nw=nw, wstep=wstep, oversampling=8,
                                                  support=32, use_aaf=False, maxsupport=512)
elapsed = time.time() - start
print("Elapsed time = ", elapsed, "sec")

#start = time.time()
#gcfcf_wt = create_awterm_convolutionfunction(model, make_pb=None, nw=nw, wstep=wstep, oversampling=8,
#                                                  support=32, use_aaf=False, maxsupport=512, wtowers=True)
#wtkern_invert = gcf2wkern2(gcfcf_wt)
#elapsed = time.time() - start
#print("Elapsed time = ", elapsed, "sec")

#wtkern_predict = gcf2wkern2(gcfcf, conjugate=True)


# In[9]:


# W-proj invert_wt results

# In[10]:

# In[17]:


# W-proj invert_2d results
start = time.time()
#dirty_wt,_ = invert_wt(blockvis, model, normalize=True, wtkern=wtkern_invert)
dirty_2d,_ = invert_2d(blockvis, model, dopsf=False, normalize=True)
elapsed = time.time() - start
print("Elapsed time = ", elapsed, "sec")


plt.rcParams['figure.figsize'] = 10, 10
show_image(dirty_2d, chan=1)
plt.savefig("dirty_invert_2d.png")

