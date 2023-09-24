#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6


__kernel void sum(const unsigned int n, __global int* arr, __global int* sum){
    const unsigned int gid = get_global_id(0);
    const unsigned int gsize = get_global_size(0);
    const unsigned int values_per_workitem = (n + gsize - 1) / gsize;

    if(gid == 0)
        *sum = 0;

    unsigned int item_sum = 0;
    for(unsigned int i = 0; i < values_per_workitem; i++){
        unsigned int idx = gid * values_per_workitem + i;
        if(idx < n){
            item_sum += arr[idx];
        }
    }
    atomic_add(sum, item_sum);
}