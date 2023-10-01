#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

__kernel void matrix_multiplication(__global float* A, __global float* B, __global float* C,
                                      unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0;
    for(int k = 0; k < K; k++){
        sum += A[i*K + k] * B[k*N + j];
    }

    C[i*N + j] = sum;
}