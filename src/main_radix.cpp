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

#define LOG_GROUP_SIZE 3
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
    VEC<gpu::gpu_mem_32u> buffers;
    uint size = N;
    while (size > 1) {
        gpu::gpu_mem_32u gpu_arr;
        gpu_arr.resizeN(size);
        buffers.push_back(std::move(gpu_arr));
        size = (size + GROUP_SIZE - 1) / GROUP_SIZE;
    }
    gpu::gpu_mem_32u gpu_arr;
    gpu_arr.resizeN(1);
    buffers.push_back(std::move(gpu_arr));
    return buffers;
}

/*
 Принимает заготовленные буферы на видеокарте, где верхний слой (buffers[0]) заполнен исходным массивом.
 После выполнения верхний слой оказывается заполнен префиксными суммами.
*/
void run_gpu_prefix_sum(
        VEC<gpu::gpu_mem_32u>& buffers,
        ocl::Kernel& prefix_sum_forward,
        ocl::Kernel& prefix_sum_backward
        )
{
    const uint N = buffers[0].size() / sizeof(uint);
    const int LEVELS = (int)buffers.size();

    const uint work_group_size = GROUP_SIZE / 2;
    const uint work_groups = (N + GROUP_SIZE - 1) / GROUP_SIZE;
    const auto ws = gpu::WorkSize(work_group_size, work_group_size * work_groups);

    for(int i = 0; i < LEVELS - 1; i++){
        std::cout << "forward, level " << i << std::endl;
        print_gpu_array(buffers[i]);
        print_gpu_array(buffers[i+1]);
        uint size_i = buffers[i].size() / sizeof(uint);
        prefix_sum_forward.exec(ws, buffers[i], buffers[i + 1], size_i);
    }

    for(int i = LEVELS - 2; i >= 0; i--){
        std::cout << "backward, level " << i << std::endl;
        print_gpu_array(buffers[i]);
        print_gpu_array(buffers[i+1]);
        uint size_i = buffers[i].size() / sizeof(uint);
        prefix_sum_backward.exec(ws, buffers[i], buffers[i + 1], size_i);
    }
}

/*
 Вот так можно считать префиксную сумму
 */
VEC<uint> gpu_prefix_sum(const VEC<uint>& arr){
    // компилируем кернелы
    ocl::Kernel prefix_sum_forward(radix_kernel, radix_kernel_length, "prefix_sum_forward");
    prefix_sum_forward.compile();
    ocl::Kernel prefix_sum_backward(radix_kernel, radix_kernel_length, "prefix_sum_backward");
    prefix_sum_backward.compile();

    // заготавливаем буферы нужных размеров на видеокарте
    const uint N = arr.size();
    auto buffers = create_buffers_for_prefix_sum(N);

    // помещаем данные в верхний слой
    buffers[0].writeN(arr.data(), N);

    // запускаем прямой и обратные проход префиксной суммы
    run_gpu_prefix_sum(buffers, prefix_sum_forward, prefix_sum_backward);

    // читаем из верхнего слоя готовые префиксные суммы
    VEC<uint> res(N);
    buffers[0].readN(res.data(), N);

    return res;
}



template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


void test_radix_gpu(VEC<uint> arr, const VEC<uint>& expected, uint benchmarkingIters){
    // кернелы для префиксной суммы
    ocl::Kernel prefix_sum_forward(radix_kernel, radix_kernel_length, "prefix_sum_forward");
    prefix_sum_forward.compile();
    ocl::Kernel prefix_sum_backward(radix_kernel, radix_kernel_length, "prefix_sum_backward");
    prefix_sum_backward.compile();

    // этот кернел извлекает k-й бит из элементов одного массива и складывает в другой
    ocl::Kernel bit_k(radix_kernel, radix_kernel_length, "bit_k");
    bit_k.compile();

    // этот кернел выполняет сортировку по k-му биту, используя предпосчитанный массив,
    // где на каждой позиции стоит количество элементов исходного массива
    // до или на этой позиции с единичным k-м битом
    ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
    radix.compile();


    const uint N = arr.size();

    // в этом буфере будем поразрядно сортировать массив
    gpu::gpu_mem_32u arr_gpu;
    arr_gpu.resizeN(N);

    // заготавливаем буферы для префиксной суммы размера N на видеокарте
    auto buffers = create_buffers_for_prefix_sum(N);

    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        arr_gpu.writeN(arr.data(), N);
//        print_gpu_array(arr_gpu);
        t.restart();
        for(uint bit = 0; bit < 32; bit++){
            std::cout << "----- k = " << bit << " ----------------------\n";
            // помещаем данные в верхний слой
            bit_k.exec(gpu::WorkSize(128, N), arr_gpu, buffers[0], bit);
            std::cout << "arr and k-th bit in arr:\n";
            print_gpu_array(arr_gpu);
            print_gpu_array(buffers[0]);

            // запускаем прямой и обратные проход префиксной суммы
            run_gpu_prefix_sum(buffers, prefix_sum_forward, prefix_sum_backward);
            std::cout << "prefix sum:\n";
            print_gpu_array(buffers[0]);

            radix.exec(gpu::WorkSize(128, N), arr_gpu, buffers[0], N,bit);
            std::cout << "array sorted by k-th bit:\n";
            print_gpu_array(arr_gpu);
        }
        t.nextLap();
    }
    std::cout << "GPU radix sort: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU radix sort: " << (N / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

    VEC<uint> arr_sorted(N);
    arr_gpu.readN(arr_sorted.data(), N);

    //print_array(arr_sorted);
    //print_array(expected);

    // Проверяем корректность результатов
    for (int i = 0; i < N; ++i) {
        EXPECT_THE_SAME(arr_sorted[i], expected[i], "GPU results should be equal to CPU results!");
    }
}


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
//    unsigned int N = 4*1024;
//    std::vector<unsigned int> arr(N, 0);
//    FastRandom r(12345);
//    for (unsigned int i = 0; i < N; ++i) {
//        arr[i] = N - i; // (unsigned int) r.next(0, std::numeric_limits<int>::max());
//    }
    VEC<uint> arr = {1,4,3,5,6,2,8,7};
    uint N = arr.size();
    std::cout << "Data generated for N=" << N << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = arr;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (N / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    test_radix_gpu(arr, cpu_sorted, benchmarkingIters);

    return 0;
}
