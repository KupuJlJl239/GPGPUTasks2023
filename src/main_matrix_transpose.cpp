#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int X = 1024;
    unsigned int Y = 1024;

    std::vector<float> as(X * Y, 0);
    std::vector<float> as_t(X * Y, 0);

    FastRandom r(X + Y);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for X=" << X << ", Y=" << Y << std::endl;


    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(X * Y);
    as_t_gpu.resizeN(Y * X);

    as_gpu.writeN(as.data(), X * Y);

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose");
    matrix_transpose_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // Для этой задачи естественнее использовать двухмерный NDRange. Чтобы это сформулировать
            // в терминологии библиотеки - нужно вызвать другую вариацию конструктора WorkSize.
            // В CLion удобно смотреть какие есть вариант аргументов в конструкторах:
            // поставьте каретку редактирования кода внутри скобок конструктора WorkSize -> Ctrl+P -> заметьте что есть 2, 4 и 6 параметров
            // - для 1D, 2D и 3D рабочего пространства соответственно
            matrix_transpose_kernel.exec(gpu::WorkSize(16, 16, X, Y), as_gpu, as_t_gpu, X, Y);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << X * Y /1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
    }

    as_t_gpu.readN(as_t.data(), X * Y);

    // Проверяем корректность результатов
    for (int j = 0; j < X; ++j) {
        for (int i = 0; i < Y; ++i) {
            float a = as[j * Y + i];
            float b = as_t[i * X + j];
            if (a != b) {
                std::cerr << "Not the same!" << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
