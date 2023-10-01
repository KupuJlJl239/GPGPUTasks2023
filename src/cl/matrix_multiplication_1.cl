#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

#define TILE_SIZE 16
__kernel void matrix_multiplication(__global const float* as, __global const float* bs, __global float* cs,
                                    const unsigned int M, const unsigned int K, const unsigned int N)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int _i = get_local_id(0);
    const int _j = get_local_id(1);
    __local float TileA[TILE_SIZE * TILE_SIZE];
    __local float TileB[TILE_SIZE * TILE_SIZE];

#define IDX(i, j, rows, cols) ((i)*(cols) + (j))
#define A(i,j) as[IDX(i,j,M,K)]
#define B(i,j) bs[IDX(i,j,K,N)]
#define C(i,j) cs[IDX(i,j,M,N)]
#define tileA(i,j) TileA[(i)*TILE_SIZE + ((i)+(j))%TILE_SIZE]
#define tileB(i,j) TileB[(i)*TILE_SIZE + ((i)+(j))%TILE_SIZE]

    float sum = 0;
    for(int tile = 0; tile * TILE_SIZE < K; tile++){
        tileA(_i, _j) = A(i, _j + TILE_SIZE*tile);
        tileB(_i, _j) = B(_i + TILE_SIZE*tile, j);
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int _k = 0; _k < TILE_SIZE; _k++){
            sum += tileA(_i, _k) * tileB(_k, _j);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C(i,j) = sum;
}