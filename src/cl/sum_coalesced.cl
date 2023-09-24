#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6


__kernel void sum(const unsigned int n, __global int* arr, __global int* sum){
    const unsigned int gid = get_global_id(0);
    const unsigned int gsize = get_global_size(0);

    if(gid == 0)
        *sum = 0;

    unsigned int item_sum = 0;
    unsigned int item_idx = gid;  // соседние item-ы обращаются к соседним элементам
    while(item_idx < n){
        item_sum += arr[item_idx];
        item_idx += gsize;
    }
    atomic_add(sum, item_sum);
}