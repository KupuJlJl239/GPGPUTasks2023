#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/h/matrix_multiplication_0_cl.h"
#include "cl/h/matrix_multiplication_1_cl.h"
#include "cl/h/matrix_multiplication_2_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <functional>

#define VEC std::vector
using uint = unsigned int;


void benchmark(const char* name, int times, float gflops, std::function<void()> f ){
    timer t;
    for (int iter = 0; iter < times; ++iter) {
        f();
        t.nextLap();
    }
    std::cout << name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << name << ": " << gflops/t.lapAvg() << " GFlops" << std::endl;
}

void check_result(const VEC<float>& result, const VEC<float>& expected){
    assert(result.size() == expected.size());
    size_t size = result.size();

    double diff_sum = 0;
    for (int i = 0; i < size; ++i) {
        double a = result[i];
        double b = expected[i];
        if (a != 0.0 || b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
    }

    double diff_avg = diff_sum / size;
    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
    }
}


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int M = 64;
    unsigned int K = 64;
    unsigned int N = 64;
    const float gflops = ((float) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

    std::vector<float> A(M*K, 0);
    std::vector<float> B(K*N, 0);
    std::vector<float> result_cpu(M*N, 0);

    FastRandom r(M+K+N);
    for (float & a : A) {
        a = r.nextf();
    }
    for (float & b : B) {
        b = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << std::endl;

    auto run_cpu= [&](){
        for (int j = 0; j < M; ++j) {
            for (int i = 0; i < N; ++i) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[j * K + k] * B[k * N + i];
                }
                result_cpu[j * N + i] = sum;
            }
        }
    };

    benchmark("CPU", benchmarkingIters, gflops, run_cpu);


    VEC<float> result(M*N, 0);
    const VEC<float> expected = result_cpu;

    gpu::gpu_mem_32f A_gpu, B_gpu, result_gpu;
    A_gpu.resizeN(M*K);
    B_gpu.resizeN(K*N);
    result_gpu.resizeN(M*N);

    A_gpu.writeN(A.data(), M*K);
    B_gpu.writeN(B.data(), K*N);


    auto test_matrix_mult_gpu =
            [&](const char* name, const char* code, size_t length, const char* k_name)
    {
        ocl::Kernel matrix_multiplication_kernel(code, length, k_name);
        matrix_multiplication_kernel.compile();

        auto work_size = gpu::WorkSize(16, 16, M, N);
        auto run = [&](){
            matrix_multiplication_kernel.exec(work_size, A_gpu, B_gpu, result_gpu, M, K, N);
        };

        benchmark(name, benchmarkingIters, gflops, run);

        result_gpu.readN(result.data(), M*N);
        check_result(result, expected);
    };

    test_matrix_mult_gpu("0", matrix_multiplication_0, matrix_multiplication_0_length, "matrix_multiplication");
    test_matrix_mult_gpu("1", matrix_multiplication_1, matrix_multiplication_1_length, "matrix_multiplication");




    return 0;
}
