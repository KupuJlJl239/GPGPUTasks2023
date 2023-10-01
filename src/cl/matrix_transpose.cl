#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

#define TILE_SIZE 16
__kernel void matrix_transpose(__global float* in_matrix, __global float* out_matrix, unsigned int X, unsigned int Y)
{
    __local float tile[TILE_SIZE*TILE_SIZE];
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);
    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

// tile[ LOCAL_IDX(local_x, local_y) ], tile[ LOCAL_IDX(local_y, local_x) ] - обращения без bank конфликтов
#define LOCAL_IDX(x, y) ((x+y)%TILE_SIZE + TILE_SIZE*y)

    // 1) копируем из глобальной памяти
    tile[ LOCAL_IDX(local_x, local_y) ] = in_matrix[global_x + X*global_y];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2) записываем результат
    int out_x = local_x + group_y * TILE_SIZE;
    int out_y = local_y + group_x * TILE_SIZE;
    if(out_x < Y && out_y < X)
        out_matrix[ out_x + Y*out_y] = tile[ LOCAL_IDX(local_y, local_x) ];
}