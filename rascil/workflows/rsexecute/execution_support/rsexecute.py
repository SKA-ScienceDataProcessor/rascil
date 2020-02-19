""" Wrap dask such that with the same code Dask.delayed can be replaced by immediate calculation

"""

__all__ = ['rsexecute', 'get_dask_client', '_rsexecutebase']

import os
import logging
import time

from tabulate import tabulate

from dask import delayed, optimize
from dask.distributed import wait
from distributed import Client, LocalCluster

log = logging.getLogger("logger")

# Support daliuge's delayed function, make it fail if not available but used
try:
    from dlg import delayed as dlg_delayed
    from dlg.dask_emulation import compute as dlg_compute
except ImportError:
    def dlg_delayed(*args, **kwargs):
        raise Exception("daliuge is not available")
    def dlg_compute(*args, **kwargs):
        pass

log = logging.getLogger('logger')


def get_dask_client(timeout=30, n_workers=None, threads_per_worker=1, processes=True, create_cluster=True,
                    memory_limit=None, local_dir='.', with_file=False,
                    scheduler_file='./scheduler.json',
                    dashboard_address=':8787'):
    """ Get a Dask.distributed Client to be used in rsexecute

    The default operation of rsexecute.set_client is to create a set of workes on one node. Hence if you
    want to use a cluster it is necessary to use get_dask_client.

    The environment variable RASCIL_DASK_SCHEDULER is interpreted as pointing to the Dask distributed scheduler.
    and a client using that scheduler is returned. Otherwise a client for a LocalCluster is created.

    :param timeout: Time out for creation (30s)
    :param n_workers: Number of workers (cores available)
    :param threads_per_worker: 1
    :param processes: Use processes instead of threads (True)
    :param create_cluster: Create a LocalCluster (True)
    :param memory_limit: Memory limit per worker (bytes e.g. 8e9) (None)
    :param scheduler_file: Scheduler file for Dask ('./scheduler.json')
    :param dashboard_address: Port used for diagnostics (':8787')
    :return: Dask client
    """
    scheduler = os.getenv('RASCIL_DASK_SCHEDULER', None)
    if scheduler is not None:
        print("Creating Dask Client using externally defined scheduler")
        c = Client(scheduler, timeout=timeout)
    elif with_file:
        print("Creating Dask Client using externally defined scheduler in file  %s" % scheduler_file)
        c = Client(scheduler_file=scheduler_file, timeout=timeout)

    elif create_cluster:
        if n_workers is not None:
            if memory_limit is not None:
                cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                                       memory_limit=memory_limit,
                                       dashboard_address=dashboard_address)
            else:
                cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                                       dashboard_address=dashboard_address)
        else:
            if memory_limit is not None:
                cluster = LocalCluster(threads_per_worker=threads_per_worker, processes=processes,
                                       memory_limit=memory_limit,
                                       dashboard_address=dashboard_address)
            else:
                cluster = LocalCluster(threads_per_worker=threads_per_worker, processes=processes,
                                       dashboard_address=dashboard_address)

        print("Creating LocalCluster and Dask Client")
        c = Client(cluster)
    else:
        c = Client(threads_per_worker=threads_per_worker, processes=processes,
                   memory_limit=memory_limit, local_dir=local_dir)

    addr = c.scheduler_info()['address']
    services = c.scheduler_info()['services']
    if 'bokeh' in services.keys():
        bokeh_addr = 'http:%s:%s' % (addr.split(':')[1], services['bokeh'])
        print('Diagnostic pages available on port %s' % bokeh_addr)
    if 'dashboard' in services.keys():
        db_addr = 'http:%s:%s' % (addr.split(':')[1], services['dashboard'])
        print('Diagnostic pages available on port %s' % db_addr)
    return c


def get_nodes():
    """ Get the nodes being used

    The environment variable RASCIL_HOSTFILE is interpreted as file containing the nodes

    :return: List of strings
    """
    hostfile = os.getenv('RASCIL_HOSTFILE', None)
    if hostfile is None:
        print("No hostfile specified")
        return None

    import socket
    with open(hostfile, 'r') as file:
        nodes = [line.replace('\n', '') for line in file.readlines()]
        print("Nodes being used are %s" % nodes)
        nodes = [socket.gethostbyname(node) for node in nodes]
        print("Nodes IPs are %s" % nodes)
        return nodes


def findNodes(c):
    """ Find Nodes being used for this Client

    """
    return [c.scheduler_info()['workers'][name]['host'] for name in c.scheduler_info()['workers'].keys()]


class _rsexecutebase():
    """ Initialise rsexecute framework

    A singleton of this class is created and is available globally as rsexecute. Hence it is not necessary to
    declare an instance of _rsexecutebase.

    For example::

        from rascil.workflows import continuum_imaging_list_rsexecute_workflow, rsexecute
        rsexecute.set_client(use_dask=True, threads_per_worker=1,
            memory_limit=32 * 1024 * 1024 * 1024, n_workers=8,
            local_dir=dask_dir, verbose=True)
        continuum_imaging_list = continuum_imaging_list_rsexecute_workflow(vis_list,
            model_imagelist=model_list,
            context='wstack', vis_slices=51,
            scales=[0, 3, 10], algorithm='mmclean',
            nmoment=3, niter=1000,
            fractional_threshold=0.1, threshold=0.1,
            nmajor=5, gain=0.25,
            psf_support=64)

        deconvolved_list, residual_list, restored_list = rsexecute.compute(continuum_imaging_list,
            sync=True)

    :param use_dask: Use dask (True)
    :param use_dlg: Use daluige (False)
    :param verbose: Be verbose in printing messages
    :param optimize: Optimize if using dask (True)
    """
    _instance = None

    def __init__(self, use_dask=True, use_dlg=False, verbose=False, optimize=True):
        """ Initialise rsexecute framework

        A singleton of this class is created and is available globally as rsexecute

        :param use_dask: Use dask (True)
        :param use_dlg: Use daluige (False)
        :param verbose: Be verbose in printing messages
        :param optimize: Optimize if using dask (True)
        """
        if bool(use_dask) and bool(use_dlg):
            raise ValueError('use_dask and use_dlg cannot be specified together')
        self._set_state(use_dask, use_dlg, None, verbose, optimize)

    def _set_state(self, use_dask, use_dlg, client, verbose, optimize):
        self._using_dask = use_dask
        self._using_dlg = use_dlg
        self._client = client
        self._verbose = verbose
        self._optimize = optimize

    def execute(self, func, *args, **kwargs):
        """ Wrap for immediate or deferred execution

        Passes through if dask is not being used

        :param args:
        :param kwargs:
        :return: delayed func or func
        """
        if self._using_dask:
            return delayed(func, *args, **kwargs)
        elif self._using_dlg:
            return dlg_delayed(func, *args, **kwargs)
        else:
            return func

    def type(self):
        """ Get the name of the execution system

        :return:
        """
        if self._using_dask:
            return 'dask'
        elif self._using_dlg:
            return 'daliuge'
        else:
            return 'function'

    def set_client(self, client=None, use_dask=True, use_dlg=False, verbose=False, optim=True, **kwargs):
        """Set the Dask/DALiuGE client to be used

        If you want to customise the Client or use an externally defined Scheduler use get_dask_client and pass it in.

        :param use_dask: Use Dask?
        :param client: If None and use_dask is True, a client will be created otherwise the client is None
        :param use_dlg: Use Daliuge to execute graphs?
        :param verbose: Be verbose in output
        :param optim: Use dask.optimize via rsexecute.optimize function.
        :return:
        """
        if bool(use_dask) and bool(use_dlg):
            raise ValueError('use_dask and use_dlg cannot be specified together')

        if isinstance(self._client, Client):
            print("Removing existing client")
            self.client.close()

        if use_dask:
            client = client or Client(**kwargs)
            assert isinstance(client, Client)
            self._set_state(True, False, client, verbose, optim)
            self._client.profile()
            self._client.get_task_stream()
            self.start_time = time.time()

        elif use_dlg:
            self._set_state(False, True, client, verbose, optim)
        else:
            self._set_state(False, False, None, verbose, optim)
        if self._verbose:
            print('rsexecute.set_client: defined Dask Client')

    def compute(self, value, sync=False):
        """Get the actual value

        If not using dask then this returns the value directly since it already is computed
        If using dask and sync=True then this waits and resturns the actual wait.
        If using dask and sync=False then this returns a future, on which you will need to call .result()

        :param value:
        :param sync: Return synchronously? (False)
        :return:
        """
        if self._using_dask:
            start = time.time()
            if self.client is None:
                return value.compute()
            else:
                future = self.client.compute(value, sync=sync)
                wait(future)
                if self._verbose:
                    duration = time.time() - start
                    log.debug("rsexecute.compute: Execution using Dask took %.3f seconds" % duration)
                    print("rsexecute.compute: Execution using Dask took %.3f seconds" % duration)
                return future
        elif self._using_dlg:
            kwargs = {'client': self._client} if self._client else {}
            return dlg_compute(value, **kwargs)
        else:
            return value

    def persist(self, graph, **kwargs):
        """Persist graph data on workers

        The graphs are placed on the workers but not computed

        No-op if not using_dask

        :param graph:
        :return:
        """
        if self.using_dask and self.client is not None:
            return self.client.persist(graph, **kwargs)
        else:
            return graph

    def scatter(self, graph, **kwargs):
        """Scatter graph data to workers

        The data are placed on the workers

        No-op if not using_dask
        :param graph:
        :return:
        """
        if self.using_dask and self.client is not None:
            return self.client.scatter(graph, **kwargs)
        else:
            return graph

    def gather(self, graph):
        """Gather graph from workers

        The data are gathered from the workers

        No-op if not using_dask

        :param graph:
        :return:
        """
        if self.using_dask and self.client is not None:
            return self.client.gather(graph)
        else:
            return graph

    def run(self, func, *args, **kwargs):
        """ Run a function on the client

        :param func:
        :return:
        """
        if self.using_dask:
            return self.client.run(func, *args, **kwargs)
        else:
            return func

    def optimize(self, *args, **kwargs):
        """ Run Dask optimisation of graphs

        Only does something when using dask

        :param args: for Dask.optimize
        :param kwargs: for Dask.optimize
        :return:
        """
        if self.using_dask and self._optimize:
            return optimize(*args, **kwargs)[0]
        else:
            return args[0]

    def close(self):
        """ Close the client

        """
        if self._using_dask and isinstance(self._client, Client):
            if self._verbose:
                print('rsexcute.close: closed down Dask Client')
            if self._client.cluster is not None:
                self._client.cluster.close()
            self._client.close()
            self._client = None

    def init_statistics(self):
        """ Initialise the profile and task stream info

        rsexecute can save the Dask profile and Task Stream information for later saving

        :return:
        """
        self.start_time = time.time()
        if self._using_dask:
            self._client.profile()
            self._client.get_task_stream()

    def save_statistics(self, name='dask'):
        """ Save the statistics to html files

        rsexecute can save the Dask profile and Task Stream information for later saving. This
        saves the current statistics to html files.

        :param name: prefix to name e.g. dask
        """

        if self._using_dask:
            task_stream, graph = self.client.get_task_stream(plot='save',
                                                             filename="%s_task_stream.html" % name)
            self.client.profile(plot='save', filename="%s_profile.html" % name)

            def print_ts(ts):
                log.info("Processor time used in each function")
                summary = {}
                number = {}
                for t in ts:
                    name = t['key'].split('-')[0]
                    elapsed = t['startstops'][0]['stop'] - t['startstops'][0]['start']
                    if name not in summary.keys():
                        summary[name] = elapsed
                        number[name] = 1
                    else:
                        summary[name] += elapsed
                        number[name] += 1
                total = 0.0
                for key in summary.keys():
                    total += summary[key]
                table = []
                headers = ["Function", "Time (s)", "Per cent", "Number calls"]
                for key in summary.keys():
                    table.append([key, "{0:.3f}".format(summary[key]), "{0:.2f}".format(100.0 * summary[key] / total),
                                  number[key]])
                log.info("\n" + tabulate(table, headers=headers))
                duration = time.time() - self.start_time
                speedup = (total / duration)
                log.info("Total processor time {0:.3f} (s), total wallclock time {1:.3f} (s), speedup {2:.2f}".
                      format(total, duration, speedup))

            try:
                print_ts(task_stream)
            except  ValueError:
                log.warning("Dask task stream is unintelligible")

    @property
    def client(self):
        """ Client being used

        :return: client
        """
        return self._client

    @property
    def using_dask(self):
        """ Is dask being used?

        :return:
        """
        return self._using_dask

    @property
    def using_dlg(self):
        """ Is daluige being used?

        :return:
        """
        return self._using_dlg

    @property
    def optimizing(self):
        """ Is Dask optimisation being performed?

        :return:
        """
        return self._optimize


def rsexecutebase(*args, **kwargs):
    if _rsexecutebase._instance is None:
        _rsexecutebase._instance = _rsexecutebase(*args, **kwargs)
    return _rsexecutebase._instance


# Any new rsexecute created by import of this file points to the only _rsexecutebase
rsexecute = rsexecutebase(use_dask=True)


