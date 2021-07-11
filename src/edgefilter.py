# custom data structure to hold the state of a Canny edge filter
class EdgeFilter:

    def __init__(self, KernelSize=None, erodeIter=None, dilateIter=None, Canny1=None, Canny2=None):
        self.KernelSize = KernelSize
        self.erodeIter = erodeIter
        self.dilateIter = dilateIter
        self.Canny1 = Canny1
        self.Canny2 = Canny2
