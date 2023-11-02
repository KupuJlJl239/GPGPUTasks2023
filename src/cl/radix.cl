#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

#define uint unsigned int

// число элементов для которых считаем префиксную сумму локально
// размер рабочей группы вдвое меньше
#define LOG_GROUP_SIZE 3
#define GROUP_SIZE (1 << LOG_GROUP_SIZE)

// arr имеет длину ровно GROUP_SIZE
inline void local_prefix_sum(__local uint *arr){
    const uint lid = get_local_id(0);
    for(int i = 0; i < LOG_GROUP_SIZE; i++){
        uint base = ((lid >> i) << (i+1)) + (1 << i);  // стёрли последние i битов, а затем перед ними вставили 1
        uint rest = lid % (1 << i);  // последние i битов
        uint from_idx = base - 1;
        uint to_idx = base + rest;
        arr[to_idx] += arr[from_idx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// считает префиксные суммы каждой группы в массиве, где:
// первый элемент имеет индекс 1 << (LOG_GROUP_SIZE * level) - 1
// и элементы отстоят на 1 << (LOG_GROUP_SIZE * level)
// с помощью этой функции строим дерево частичных префиксных сумм
__kernel void prefix_sum_forward(__global uint *arr, __global uint *sums, uint N) {
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint grid = get_group_id(0);


    __local uint local_arr[GROUP_SIZE];

    local_arr[2 * lid] = 2 * gid < N? arr[2 * gid] : 0;
    local_arr[2 * lid + 1] = 2 * gid + 1 < N? arr[2 * gid + 1] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // после этого в local_arr лежат не элементы массива, а префиксные суммы
    local_prefix_sum(local_arr);

    if(2 * gid < N)
        arr[2 * gid] = local_arr[2 * lid];
    if(2 * gid + 1 < N)
        arr[2 * gid + 1] = local_arr[2 * lid + 1];

    if(lid == 0 && N > 1)
        sums[grid] = local_arr[GROUP_SIZE - 1];
}

__kernel void prefix_sum_backward(__global uint *arr, __global uint *sums, uint N) {
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint grid = get_group_id(0);

    uint sum = grid > 0? sums[grid - 1] : 0;

    if(2 * gid < N)
        arr[2 * gid] += sum;
    if(2 * gid + 1 < N)
        arr[2 * gid + 1] += sum;
}

__kernel void bit_k(__global uint *arr, __global uint *bits, uint bit_number){
    uint gid = get_global_id(0);
    bits[gid] = (arr[gid] >> bit_number) & 1;
}


__kernel void radix(__global uint *arr, __global const uint *prefix_ones, const uint N, uint bit_number) {
    uint gid = get_global_id(0);
    uint arr_i = gid < N ? arr[gid] : 0;
    char bit = (arr_i >> bit_number) & 1;

    uint ones_before = 0 < gid && gid < N ? prefix_ones[gid - 1] : 0;
    uint zeros_before = gid - ones_before;
    uint ones_total = prefix_ones[N - 1];
    uint zeros_total = N - ones_total;

    uint idx = bit ? zeros_total + ones_before : zeros_before;

//    if(gid < N && (arr_i < 0 || 128 < arr_i) )
//        printf("gid=%d, idx=%d, arr_i=%d\n", gid, idx, arr_i);

    barrier(CLK_GLOBAL_MEM_FENCE);
    if(gid < N)
        arr[idx] = arr_i;
}
