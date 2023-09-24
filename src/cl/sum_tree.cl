#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

#define MAX_WORKGROUP_SIZE 64
__kernel void sum(const unsigned int n, __global int* arr, __global int* sum){
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int wsize = get_local_size(0);
    const unsigned int gsize = get_local_size(0);

    if(gid == 0)
        *sum = 0;

    __local unsigned int buf[MAX_WORKGROUP_SIZE];
    buf[lid] = gid < n? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // На самом деле в buf считаются префиксные суммы
    int shift = 1;
    while(shift < gsize){
        int idx = (int)lid - shift;
        if(idx >= 0){
            buf[lid] += buf[idx];
        }
        shift *= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0){
        atomic_add(sum, buf[gsize-1]);
    }
}