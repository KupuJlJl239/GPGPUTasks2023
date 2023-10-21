#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

#define TILE_SIZE 16
#define ITEM_WORK 4
__kernel void matrix_multiplication(__global const float* as, __global const float* bs, __global float* cs,
                                    const unsigned int M, const unsigned int K, const unsigned int N)
{
    const int i = get_global_id(1)*ITEM_WORK;
    const int j = get_global_id(0);
    const int _i = get_local_id(1)*ITEM_WORK;
    const int _j = get_local_id(0);
    __local float TileA[TILE_SIZE * TILE_SIZE];
    __local float TileB[TILE_SIZE * TILE_SIZE];

#define IDX(i, j, rows, cols) ((i)*(cols) + (j))
#define A(i,j) as[IDX(i,j,M,K)]
#define B(i,j) bs[IDX(i,j,K,N)]
#define C(i,j) cs[IDX(i,j,M,N)]
#define tileA(i,j) TileA[(i)*TILE_SIZE + ((i)+(j))%TILE_SIZE]
#define tileB(i,j) TileB[(i)*TILE_SIZE + ((i)+(j))%TILE_SIZE]

    float sum[ITEM_WORK];
    for(int w = 0; w < ITEM_WORK; w++)
        sum[w] = 0;

    for(int tile = 0; tile * TILE_SIZE < K; tile++){
        for(int w = 0; w < ITEM_WORK; w++) {
            tileA(_i + w, _j) = A(i + w, _j + TILE_SIZE * tile);
            tileB(_i + w, _j) = B(_i + w + TILE_SIZE * tile, j);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int _k = 0; _k < TILE_SIZE; _k++){
            float b = tileB(_k, _j);
            for(int w = 0; w < ITEM_WORK; w++)
                sum[w] += tileA(_i + w, _k) * b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for(int w = 0; w < ITEM_WORK; w++)
        C(i+w,j) = sum[w];
}



//#ifdef __CLION_IDE__
//    #include <libgpu/opencl/cl/clion_defines.cl>
//    #include <libgpu/opencl/cl/common.cl>
//#endif
//
//#line 6
//
//#define TILE_SIZE 16
//#define ITEM_WORK 4
//__kernel void matrix_multiplication(__global const float* as, __global const float* bs, __global float* cs,
//                                    const unsigned int M, const unsigned int K, const unsigned int N)
//{
//    const int i = get_global_id(1);
//    const int j = get_global_id(0)*ITEM_WORK;
//    const int _i = get_local_id(1);
//    const int _j = get_local_id(0)*ITEM_WORK;
//    const int g_row = get_group_id(1)*TILE_SIZE;
//    const int g_col = get_group_id(0)*TILE_SIZE;
//
//    __local float TileA[TILE_SIZE * TILE_SIZE];
//    __local float TileB[TILE_SIZE * TILE_SIZE];
//
//#define IDX(i, j, rows, cols) ((i)*(cols) + (j))
//#define A(i,j) as[IDX(i,j,M,K)]
//#define B(i,j) bs[IDX(i,j,K,N)]
//#define C(i,j) cs[IDX(i,j,M,N)]
//#define tileA(i,j) TileA[(i)*TILE_SIZE + ((i)+(j))%TILE_SIZE]
//#define tileB(i,j) TileB[(i)*TILE_SIZE + ((i)+(j))%TILE_SIZE]
//
//    float sum[ITEM_WORK];
//    for(int w = 0; w < ITEM_WORK; w++)
//        sum[w] = 0;
//
//    for(int tile = 0; tile * TILE_SIZE < K; tile++){
//        for(int w = 0; w < ITEM_WORK; w++) {
//            tileA(_i, _j + w) = A(i, _j + TILE_SIZE * tile + w);
//            tileB(_i, _j + w) = B(_i + TILE_SIZE * tile, j + w);
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//
//        for(int _k = 0; _k < TILE_SIZE; _k++){
//            float a = tileA(_i, _k);
//            for(int w = 0; w < ITEM_WORK; w++)
//                sum[w] += a * tileB(_k, _j + w);
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//
//    for(int w = 0; w < ITEM_WORK; w++)
//        C(i,j+w) = sum[w];
//}