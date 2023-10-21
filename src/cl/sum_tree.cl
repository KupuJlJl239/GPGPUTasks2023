#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

#define WORKGROUP_SIZE 64
__kernel void sum(const unsigned int n, __global int* arr, __global int* sum){
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int gsize = get_local_size(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < n? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    int shift = 1;
    // если использовать вместо WORKGROUP_SIZE gsize, то будет в 2 раза медленнее
    while(shift < WORKGROUP_SIZE){
        int idx = (int)lid + shift;
        if((lid % (2*shift)) == 0){
            buf[lid] += buf[idx];
        }
        shift *= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0){
        atomic_add(sum, buf[0]);
    }
}