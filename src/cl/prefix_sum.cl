#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

#define uint unsigned int

__kernel void prefix_sum(__global uint* arr, uint step)
{
    const uint gid = get_global_id(0);
    if(gid & (1 << step))
        arr[gid] += arr[gid - (1 << step)];
}