'''
Custom non-daemonic Pool class
Code adapted from https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
'''
import multiprocessing
import multiprocessing.pool


# class NoDaemonProcess(multiprocessing.Process):
#     def _get_daemon(self):
#         return False
#     def _set_daemon(self, value):
#         pass
#     daemon = property(_get_daemon, _set_daemon)

# class MyPool(multiprocessing.pool.Pool):
#     Process = NoDaemonProcess
#     def __init__(self, *args, **kwargs):
#         super(MyPool, self).__init__(*args, **kwargs)

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)