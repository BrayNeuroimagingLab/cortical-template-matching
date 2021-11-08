# distutils: language = c++
import Cython

cdef extern from "getRpthreaded.cpp":
    void initThreads(double *, float *, size_t, size_t)
    void * biteR(void *)

def getRpthreaded_py(double[:,::1] T, float[:,::1] R, int voxelCount, int time):
    initThreads(&T[0,0], &R[0,0], voxelCount, time)
    return R
