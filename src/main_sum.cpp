#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Эти файлы будут сгенерированы автоматически в момент сборки
#include "cl/h/sum_base_cl.h"
#include "cl/h/sum_coalesced_cl.h"
#include "cl/h/sum_cycle_cl.h"
#include "cl/h/sum_local_mem_cl.h"
#include "cl/h/sum_tree_cl.h"


using uint = unsigned int;
#define VEC std::vector


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


void test_sum(
        const char* name,
        uint reference_sum,
        const VEC<uint>& as,
        int benchmarkingIters,
        uint (*make_sum)(const VEC<uint>& )
){
    uint n = as.size();
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        uint sum = (*make_sum)(as);
        EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
        t.nextLap();
    }
    std::cout << name << ":     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << name << ":     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
}


uint generate_array(uint n, VEC<uint>& as){
    as.clear();
    as.resize(n);
    uint reference_sum = 0;
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (uint) r.next(0, std::numeric_limits<uint>::max() / n);
        reference_sum += as[i];
    }
    return reference_sum;
}


uint gpu_sum(const VEC<uint> as, const char* compiled_kernel, size_t kernel_length, const char* kernel_name){
    const uint n = as.size();

    ocl::Kernel kernel(compiled_kernel, kernel_length, kernel_name);
    kernel.compile();

    gpu::gpu_mem_32u as_gpu, sum_gpu;
    as_gpu.resizeN(n);
    sum_gpu.resizeN(1);

    as_gpu.writeN(as.data(), n);

    kernel.exec(gpu::WorkSize(128, n),
                as_gpu, sum_gpu);

    uint sum;
    sum_gpu.readN(&sum, 1);
    return sum;
}


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    uint n = 100*1000*1000; VEC<uint> as;
    uint reference_sum = generate_array(n, as);

    std::cout << "Make sum of " << n << " elements " << benchmarkingIters << " times" << std::endl;

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    // Этот контекст после активации будет прозрачно использоваться при всех вызовах в libgpu библиотеке
    // это достигается использованием thread-local переменных, т.е. на самом деле контекст будет активирован для текущего потока исполнения
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    auto simple_cpu_sum = [](const VEC<uint>& as){
        uint sum = 0;
        for (int i = 0; i < as.size(); ++i) {
            sum += as[i];
        }
        return sum;
    };

    auto openmp_cpu_sum = [](const VEC<uint>& as){
        unsigned int sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < as.size(); ++i) {
            sum += as[i];
        }
        return sum;
    };

    auto base_gpu_sum = [](const VEC<uint>& as){
        return gpu_sum(as, sum_base_kernel, sum_base_kernel_length, "sum");
    };

    test_sum("CPU one thread", reference_sum, as, benchmarkingIters, simple_cpu_sum);
    test_sum("CPU multi thread", reference_sum, as, benchmarkingIters, openmp_cpu_sum);
    test_sum("GPU atomic add", reference_sum, as, benchmarkingIters, base_gpu_sum);

}
