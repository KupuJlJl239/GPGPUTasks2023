#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#define uint unsigned int

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 1024*1024;
    std::vector<float> as(n, 0);
    FastRandom r(0);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << ((float)n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    std::vector<float> gpu_sorted(n);
    gpu::gpu_mem_32f as_gpu, bs_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
    auto in_gpu = &as_gpu;
    auto out_gpu = &bs_gpu;
    {
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        merge.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            (*in_gpu).writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных

            for(uint step = 0; 1<<step < n; step++) {
                merge.exec(gpu::WorkSize(1, as.size()), *in_gpu, *out_gpu, step);
                std::swap(in_gpu, out_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << ((float)n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        (*in_gpu).readN(gpu_sorted.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        // std::cout << "  " << as[i] << " " << gpu_sorted[i] << " " << cpu_sorted[i] << "\n";
        EXPECT_THE_SAME(gpu_sorted[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
