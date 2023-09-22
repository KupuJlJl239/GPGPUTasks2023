#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results,
                        // unsigned int width, unsigned int height,  // это определяется размером рабочего пространства
                        float fromX, float fromY,
                        float sizeX, float sizeY,
                        unsigned int iters, int smoothing) {
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    int idx = get_global_id(0);
    int idy = get_global_id(1);

    int width = get_global_size(0);
    int height = get_global_size(1);

    float x0 = fromX + (idx + 0.5f) * sizeX / width;
    float y0 = fromY + (idy + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }
    float result = iter;
    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iters;
    results[idy * width + idx] = result;

}