""" Functions for calibration, including creation of gaintables, application of gaintables, and
merging gaintables.

"""

__all__ = ['gaintable_summary', 'gaintable_plot', 'qa_gaintable', 'apply_gaintable', 'append_gaintable',
           'create_gaintable_from_blockvisibility', 'create_gaintable_from_blockvisibility',
           'create_gaintable_from_rows', 'copy_gaintable']

import copy
import logging
from typing import Union

import numpy.linalg

from rascil.data_models.memory_data_models import GainTable, BlockVisibility, QA, assert_vis_gt_compatible
from rascil.data_models.polarisation import ReceptorFrame

log = logging.getLogger('logger')


def apply_gaintable(vis: BlockVisibility, gt: GainTable, inverse=False, **kwargs) -> BlockVisibility:
    """Apply a gain table to a block visibility

    The corrected visibility is::

        V_corrected = {g_i * g_j^*}^-1 V_obs

    If the visibility data are polarised e.g. polarisation_frame("linear") then the inverse operator
    represents an actual inverse of the gains.

    :param vis: Visibility to have gains applied
    :param gt: Gaintable to be applied
    :param inverse: Apply the inverse (default=False)
    :return: input vis with gains applied

    """
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis
    assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt
    
    assert_vis_gt_compatible(vis, gt)
    
    if inverse:
        log.debug('apply_gaintable: Apply inverse gaintable')
    else:
        log.debug('apply_gaintable: Apply gaintable')
    
    is_scalar = gt.gain.shape[-2:] == (1, 1)
    if is_scalar:
        log.debug('apply_gaintable: scalar gains')
    
    row_numbers = numpy.array(list(range(len(vis.time))), dtype='int')
    for row in range(gt.ntimes):
        vis_rows = numpy.abs(vis.time - gt.time[row]) < gt.interval[row] / 2.0
        vis_rows = row_numbers[vis_rows]
        if len(vis_rows) > 0:
            
            # Lookup the gain for this set of visibilities
            gain = gt.data['gain'][row]
            gainwt = gt.data['weight'][row]
            
            # The shape of the mueller matrix is
            nant, nchan, nrec, _ = gain.shape
            
            original = vis.vis[vis_rows]
            applied = copy.copy(vis.vis[vis_rows])
            appliedwt = copy.copy(vis.weight[vis_rows])
            if is_scalar:
                if inverse:
                    lgain = numpy.ones_like(gain)
                    lgain[numpy.abs(gain) > 0.0] = 1.0 / gain[numpy.abs(gain) > 0.0]
                else:
                    lgain = gain
                tlgain = lgain.T
                tclgain = numpy.conjugate(tlgain)
                
                smueller = numpy.ones([nchan, nant, nant], dtype='complex')
                for chan in range(nchan):
                    smueller[chan, :, :] = numpy.ma.outer(tlgain[0, 0, chan, :],
                                                          tclgain[0, 0, chan, :]).reshape([nant, nant])
                for sub_vis_row in range(original.shape[0]):
                    for chan in range(nchan):
                        applied[sub_vis_row, :, :, chan, 0] = \
                            original[sub_vis_row, :, :, chan, 0] * smueller[chan, :, :]
                        antantwt = numpy.outer(gainwt[:, chan, 0, 0], gainwt[:, chan, 0, 0])
                        appliedwt[sub_vis_row, :, :, chan, 0] = antantwt
                        applied[sub_vis_row, :, :, chan, 0][antantwt == 0.0] = 0.0
            else:
                
                smueller = numpy.ones([nant, nant, nchan, nrec ** 2, nrec ** 2], dtype='complex')
                has_inverse = numpy.zeros([nant, nant, nchan], dtype='bool')
                for a1 in range(vis.nants - 1):
                    for a2 in range(a1 + 1, vis.nants):
                        for chan in range(nchan):
                            if inverse:
                                try:
                                    m = numpy.kron(gain[a1, chan, :, :],
                                                   numpy.conjugate(gain[a2, chan, :, :]))
                                    smueller[a2, a1, chan] = numpy.linalg.inv(m)
                                    has_inverse[a2, a1, chan] = True
                                except numpy.linalg.linalg.LinAlgError:
                                    has_inverse[a2, a1, chan] = False
                            else:
                                smueller[a2, a1, chan] = numpy.kron(gain[a1, chan, :, :],
                                                                    numpy.conjugate(gain[a2, chan, :, :]))
                for sub_vis_row in range(original.shape[0]):
                    for a1 in range(vis.nants - 1):
                        for a2 in range(a1 + 1, vis.nants):
                            for chan in range(nchan):
                                if (not inverse) or has_inverse[a2, a1, chan]:
                                    applied[sub_vis_row, a2, a1, chan, ...] = \
                                        numpy.matmul(smueller[a2, a1, chan], original[sub_vis_row, a2, a1, chan, ...])
                                else:
                                    applied[sub_vis_row, a2, a1, chan, ...] = original[sub_vis_row, a2, a1, chan, ...]

            vis.data['vis'][vis_rows] = applied
    
    return vis


def gaintable_summary(gt: GainTable):
    """Return string summarizing the Gaintable

    :param gt: Gaintable
    :returns: string

    """
    return "%s rows, %.3f GB" % (gt.data.shape, gt.size())


def create_gaintable_from_blockvisibility(vis: BlockVisibility, timeslice=None,
                                          frequencyslice: float = None, **kwargs) -> GainTable:
    """ Create gain table from visibility.
    
    This makes an empty gain table consistent with the BlockVisibility.
    
    :param vis: BlockVisibilty
    :param timeslice: Time interval between solutions (s)
    :param frequencyslice: Frequency solution width (Hz) (NYI)
    :return: GainTable
    
    """
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis
    
    nants = vis.nants
    
    if timeslice is None or timeslice == 'auto':
        utimes = numpy.unique(vis.time)
        gain_interval = vis.integration_time
    else:
        utimes = vis.time[0] + timeslice * numpy.unique(numpy.round((vis.time - vis.time[0]) / timeslice))
        gain_interval = timeslice * numpy.ones_like(utimes)
    
    ntimes = len(utimes)
    
    #    log.debug('create_gaintable_from_blockvisibility: times are %s' % str(utimes))
    #    log.debug('create_gaintable_from_blockvisibility: intervals are %s' % str(gain_interval))
    
    ntimes = len(utimes)
    ufrequency = numpy.unique(vis.frequency)
    nfrequency = len(ufrequency)
    
    receptor_frame = ReceptorFrame(vis.polarisation_frame.type)
    nrec = receptor_frame.nrec
    
    gainshape = [ntimes, nants, nfrequency, nrec, nrec]
    gain = numpy.ones(gainshape, dtype='complex')
    if nrec > 1:
        gain[..., 0, 1] = 0.0
        gain[..., 1, 0] = 0.0
    
    gain_weight = numpy.ones(gainshape)
    gain_time = utimes
    gain_frequency = ufrequency
    gain_residual = numpy.zeros([ntimes, nfrequency, nrec, nrec])
    
    gt = GainTable(gain=gain, time=gain_time, interval=gain_interval, weight=gain_weight, residual=gain_residual,
                   frequency=gain_frequency, receptor_frame=receptor_frame, phasecentre=vis.phasecentre,
                   configuration=vis.configuration)
    
    assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt
    assert_vis_gt_compatible(vis, gt)
    
    return gt


def append_gaintable(gt: GainTable, othergt: GainTable) -> GainTable:
    """Append othergt to gt

    :param gt:
    :param othergt:
    :return: GainTable gt + othergt
    """
    assert gt.receptor_frame == othergt.receptor_frame
    gt.data = numpy.hstack((gt.data, othergt.data))
    return gt


def copy_gaintable(gt: GainTable, zero=False):
    """Copy a GainTable

    Performs a deepcopy of the data array
    """
    
    if gt is None:
        return gt
    
    assert isinstance(gt, GainTable), gt
    
    newgt = copy.copy(gt)
    newgt.data = copy.deepcopy(gt.data)
    if zero:
        newgt.data['gt'][...] = 0.0
    return newgt


def create_gaintable_from_rows(gt: GainTable, rows: numpy.ndarray, makecopy=True) \
        -> Union[GainTable, None]:
    """ Create a GainTable from selected rows

    :param gt: GainTable
    :param rows: Boolean array of row selection
    :param makecopy: Make a deep copy (True)
    :return: GainTable
    """
    
    if rows is None or numpy.sum(rows) == 0:
        return None
    
    assert len(rows) == gt.ntimes, "Length of rows does not agree with length of GainTable"
    
    assert isinstance(gt, GainTable), gt
    
    if makecopy:
        newgt = copy_gaintable(gt)
        newgt.data = copy.deepcopy(gt.data[rows])
        return newgt
    else:
        gt.data = copy.deepcopy(gt.data[rows])
        
        return gt


def qa_gaintable(gt: GainTable, context=None) -> QA:
    """Assess the quality of a gaintable

    :param gt:
    :return: QA
    """
    agt = numpy.abs(gt.gain[gt.weight > 0.0])
    pgt = numpy.angle(gt.gain[gt.weight > 0.0])
    rgt = gt.residual[numpy.sum(gt.weight, axis=1) > 0.0]
    data = {'shape': gt.gain.shape,
            'maxabs-amp': numpy.max(agt),
            'minabs-amp': numpy.min(agt),
            'rms-amp': numpy.std(agt),
            'medianabs-amp': numpy.median(agt),
            'maxabs-phase': numpy.max(pgt),
            'minabs-phase': numpy.min(pgt),
            'rms-phase': numpy.std(pgt),
            'medianabs-phase': numpy.median(pgt),
            'residual': numpy.max(rgt)
            }
    return QA(origin='qa_gaintable', data=data, context=context)


def gaintable_plot(gt: GainTable, ax, title='', value='amp', ants=None, channels=None,
                   label_max=10, min_amp=1e-5, **kwargs):
    """ Standard plot of gain table

    :param gt: Gaintable
    :param ax: matplotlib axes
    :param value: 'amp' or 'phase' or 'residual'
    :param ants: Antennas to plot
    :param channels: Channels to plot
    :param kwargs:
    :return:
    """
    if ants is None:
        ants = range(gt.nants)
    if channels is None:
        channels = range(gt.nchan)
    
    if value == "residual":
        residual = gt.residual[:, channels, 0, 0]
        ax.plot(gt.time, residual, '.')
    else:
        for ant in ants:
            if gt.configuration is not None:
                label = gt.configuration.names[ant]
            else:
                label = ''
            amp = numpy.abs(gt.gain[:, ant, channels, 0, 0])
            if value == 'amp':
                ax.plot(gt.time[amp[:, 0] > min_amp], amp[amp[:, 0] > min_amp], '.', label=label)
            else:
                angle = numpy.angle(gt.gain[:, ant, channels, 0, 0])
                ax.plot(gt.time[amp[:, 0] > min_amp], angle[amp[:, 0] > min_amp], '.', label=label)
        if gt.configuration is not None:
            if len(gt.configuration.names) < label_max:
                ax.legend()
    
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(value)
