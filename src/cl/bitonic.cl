#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

#define uint unsigned int

__kernel void bitonic(__global float* arr, uint n, uint epoch, uint step)
{
    uint gid = get_global_id(0);
    uint rank = epoch - step;

    char reverse = (gid>>epoch) & 1;
    uint i = ((gid >> rank) << (rank+1)) + (gid % (1<<rank));  // insert bit 0 to position $rank$ in $gid$
    uint j = i ^ (1<<rank); // insert bit 1 to position $rank$ in $gid$

    if(n <= i || n <= j)
        return;

    float a = arr[i];
    float b = arr[j];

    char should_swap = (a > b) ^ reverse;
    if(should_swap){
        arr[i] = b;
        arr[j] = a;
    }
}