#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

#define uint unsigned int

__kernel void merge(__global float* in, __global float* out, uint step)
{

#define CHECK_LESS(i, a, j, b) a < b || (a == b && i < j)  // a = in[i], b = in[j]

    const uint n = get_global_size(0);
    const uint idx = get_global_id(0);
    float val = in[idx];
    const uint size = 1<<step;

    uint nA = idx / size;
    uint nB = nA % 2 == 0? nA+1:nA-1;
    uint nres = idx / (2*size);
    __global float* res = &out[nres*2*size];

    uint countA = idx % size;
    uint countB; // == ???

    uint p0 = nB * size;
    uint p1 = p0 + size - 1;

    if(CHECK_LESS(idx, val, p0, in[p0])){
        countB = 0;
    }
    else if(CHECK_LESS(p1, in[p1], idx, val)){
        countB = size;
    }
    else {
        while (p1 - p0 > 1) {
            uint p = (p0 + p1) / 2;
            if (CHECK_LESS(p, in[p], idx, val))
                p0 = p;
            else
                p1 = p;
        }
        countB = p1 - nB * size;
    }
    res[countA + countB] = in[idx];
}
