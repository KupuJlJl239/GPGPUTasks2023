#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

__kernel void matrix_multiplication_0(__global float* A, __global float* B, __global float* C, 
                                      unsigned int M, unsigned int K, unsigned int N)
{
    // TODO
}