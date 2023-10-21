#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <functional>

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


void benchmark(const char* name, int times, double mflops, std::function<void()> f ){
    timer t;
    for (int iter = 0; iter < times; ++iter) {
        f();
        t.nextLap();
    }
    std::cout << name << ":     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << name << ":     " << mflops / t.lapAvg() << " millions/s" << std::endl;
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



int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    uint n = 1000*1000; VEC<uint> as;
    const uint expected_sum = generate_array(n, as);
    const double mflops = (n/1000.0/1000.0);

    std::cout << "Make sum of " << n << " elements " << benchmarkingIters << " times" << std::endl;

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    // Этот контекст после активации будет прозрачно использоваться при всех вызовах в libgpu библиотеке
    // это достигается использованием thread-local переменных, т.е. на самом деле контекст будет активирован для текущего потока исполнения
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    gpu::gpu_mem_32u as_gpu, sum_gpu;
    as_gpu.resizeN(n);
    as_gpu.writeN(as.data(), n);
    sum_gpu.resizeN(1);

    const uint zero = 0;

    auto simple_cpu_sum = [&](){
        uint sum = 0;
        for (uint a : as) {
            sum += a;
        }
        EXPECT_THE_SAME(expected_sum, sum, "CPU result should be consistent!");
    };

    auto openmp_cpu_sum = [&](){
        unsigned int sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < as.size(); ++i) {
            sum += as[i];
        }
        EXPECT_THE_SAME(expected_sum, sum, "CPU result should be consistent!");
    };

    auto test_gpu_sum = [&](const char* name,
                            const char* compiled_kernel, size_t kernel_length, const gpu::WorkSize& work_size)
    {
        ocl::Kernel kernel(compiled_kernel, kernel_length, "sum");
        kernel.compile();

        auto run = [&](){
            sum_gpu.writeN(&zero, 1);
            kernel.exec(work_size,n, as_gpu,sum_gpu);
        };

        benchmark(name, benchmarkingIters, mflops, run);

        uint sum;
        sum_gpu.readN(&sum, 1);
        EXPECT_THE_SAME(expected_sum, sum, "CPU result should be consistent!");
    };


    benchmark("CPU one thread", benchmarkingIters, mflops, simple_cpu_sum);
    benchmark("CPU multi thread", benchmarkingIters, mflops, openmp_cpu_sum);

    {
        auto work_size = gpu::WorkSize(128, as.size());
        test_gpu_sum("GPU atomic add", sum_base_kernel, sum_base_kernel_length, work_size);
    }

    {
        uint elements_per_item = 256;
        auto work_size = gpu::WorkSize(128, as.size() / elements_per_item);
        test_gpu_sum("GPU with cycle", sum_cycle_kernel, sum_cycle_kernel_length, work_size);
    }

    {
        uint elements_per_item = 256;
        auto work_size = gpu::WorkSize(128, as.size() / elements_per_item);
        test_gpu_sum("GPU with coalesced cycle", sum_coalesced_kernel, sum_coalesced_kernel_length, work_size);
    }

    {
        // число item-ов чуть больше длины массива
        uint group_size = 64;
        uint groups_count = (as.size() + group_size - 1) / group_size;
        auto work_size = gpu::WorkSize(64, group_size * groups_count);
        test_gpu_sum("GPU with local mem", sum_local_mem_kernel, sum_local_mem_kernel_length, work_size);
    }

    {
        // число item-ов чуть больше длины массива
        uint group_size = 64;
        uint groups_count = (as.size() + group_size - 1) / group_size;
        auto work_size = gpu::WorkSize(64, group_size * groups_count);
        test_gpu_sum("GPU with tree sum", sum_tree_kernel, sum_tree_kernel_length, work_size);
    }

}
