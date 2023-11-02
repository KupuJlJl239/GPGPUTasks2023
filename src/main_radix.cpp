#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cassert>

using uint = unsigned int;
#define VEC std::vector

#define LOG_GROUP_SIZE 7
#define GROUP_SIZE (1 << LOG_GROUP_SIZE)



void print_array(const VEC<uint>& arr){
    for(uint el: arr)
        std::cout << el << " ";
    std::cout << std::endl;
}

void print_gpu_array(gpu::gpu_mem_32u& gpu_arr){
    VEC<uint> arr(gpu_arr.size() / sizeof(uint));
    gpu_arr.readN(arr.data(), arr.size());
    print_array(arr);
}

VEC<gpu::gpu_mem_32u> create_buffers_for_prefix_sum(const uint N) {
    VEC<gpu::gpu_mem_32u> prefix_levels;

    uint size = N;
    while (size > 1) {
        gpu::gpu_mem_32u gpu_arr;
        gpu_arr.resizeN(size);
        prefix_levels.push_back(std::move(gpu_arr));
        size = (size + GROUP_SIZE - 1) / GROUP_SIZE;
    }
    gpu::gpu_mem_32u gpu_arr;
    gpu_arr.resizeN(1);
    prefix_levels.push_back(std::move(gpu_arr));

    return prefix_levels;
}

/*
 Принимает заготовленные буферы на видеокарте, где верхний слой (prefix_levels[0]) заполнен исходным массивом.
 После выполнения верхний слой оказывается заполнен префиксными суммами.
*/
void run_gpu_prefix_sum(
        VEC<gpu::gpu_mem_32u>& prefix_levels,
        ocl::Kernel& prefix_sum_forward,
        ocl::Kernel& prefix_sum_backward
        )
{
    const uint N = prefix_levels[0].size() / sizeof(uint);
    const int LEVELS = prefix_levels.size();
//    std::cout << LEVELS << "\n";

    for(int i = 0; i < LEVELS - 1; i++){
//        std::cout << "forward, level " << i << std::endl;
//        print_gpu_array(prefix_levels[i]);
//        print_gpu_array(prefix_levels[i+1]);

        auto ws = gpu::WorkSize(GROUP_SIZE/2, N/2);
        prefix_sum_forward.exec(ws, prefix_levels[i], prefix_levels[i + 1], prefix_levels[i].size() / sizeof(uint));
//
//        print_gpu_array(prefix_levels[i]);
//        print_gpu_array(prefix_levels[i+1]);
    }

    for(int i = LEVELS - 2; i >= 0; i--){
//        std::cout << "backward, level " << i << std::endl;
//        print_gpu_array(prefix_levels[i]);
//        print_gpu_array(prefix_levels[i+1]);

        auto ws = gpu::WorkSize(GROUP_SIZE/2, N/2);
        prefix_sum_backward.exec(ws, prefix_levels[i], prefix_levels[i + 1], prefix_levels[i].size() / sizeof(uint));
//
//        print_gpu_array(prefix_levels[i]);
//        print_gpu_array(prefix_levels[i+1]);
    }



}

VEC<uint> gpu_prefix_sum(const VEC<uint>& arr){
    // компилируем кернелы
    ocl::Kernel prefix_sum_forward(radix_kernel, radix_kernel_length, "prefix_sum_forward");
    prefix_sum_forward.compile();
    ocl::Kernel prefix_sum_backward(radix_kernel, radix_kernel_length, "prefix_sum_backward");
    prefix_sum_backward.compile();

    // заготавливаем буферы нужных размеров на видеокарте
    const uint N = arr.size();
    auto prefix_levels = create_buffers_for_prefix_sum(N);

    // помещаем данные в верхний слой
    prefix_levels[0].writeN(arr.data(), N);

    // запускаем прямой и обратные проход префиксной суммы
    run_gpu_prefix_sum(prefix_levels, prefix_sum_forward, prefix_sum_backward);

    // читаем из верхнего слоя готовые префиксные суммы
    VEC<uint> res(N);
    prefix_levels[0].readN(res.data(), N);

    return res;
}


void random_test_prefix_sum(uint n){
    std::cout << "random_test_prefix_sum, n = " << n << ": ";
    uint values_range = std::min<uint>(1023, std::numeric_limits<int>::max() / n);


    VEC<uint> arr(n, 0);
    FastRandom r(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = r.next(0, values_range);
    }
//    VEC<uint> arr(8, 1);
//    n = arr.size();

    VEC<uint> prefix_sums(n, 0);
    for (int i = 0; i < n; ++i) {
        prefix_sums[i] = arr[i];
        if (i) {
            prefix_sums[i] += prefix_sums[i-1];
        }
    }

    auto result = gpu_prefix_sum(arr);
//    print_array(prefix_sums);
//    print_array(result);
    assert(prefix_sums == result);
    std::cout << "OK" << std::endl;
}



int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    for(int n = 26785; n < 7564378; n = n * 2 + 1)
        random_test_prefix_sum(n);

}





template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int _main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }
    /*
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            // TODO
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
*/
    return 0;
}
