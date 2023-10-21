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
    if(lid == 0){
        unsigned int s = 0;

        // если использовать вместо WORKGROUP_SIZE gsize, то будет в 3 раза медленнее
        for(int i = 0; i < WORKGROUP_SIZE; i++){
            s += buf[i];
        }
        atomic_add(sum, s);
    }
}
