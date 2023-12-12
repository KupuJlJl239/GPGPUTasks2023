#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"

using uint = unsigned int;
#define VEC std::vector

#define LOG_GROUP_SIZE 3
#define GROUP_SIZE (1 << LOG_GROUP_SIZE)

#define ROUND_UP_TO(n, p) (((n+p-1)/p)*p)

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


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


VEC<gpu::gpu_mem_32u> create_buffers_for_prefix_sum(const uint N){
    VEC<gpu::gpu_mem_32u> prefix_levels;
    uint size = N;
    while(size > 1){
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

void test_gpu_prefix_sum(const VEC<uint>& arr, const VEC<uint>& expected, const uint benchmarkingIters){
//    std::cout << "Compiling kernels... ";
    ocl::Kernel prefix_sum_forward(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_forward");
    prefix_sum_forward.compile();
    ocl::Kernel prefix_sum_backward(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_backward");
    prefix_sum_backward.compile();
//    std::cout << "Ready.\n";

    const uint N = arr.size();
    VEC<gpu::gpu_mem_32u> prefix_levels = create_buffers_for_prefix_sum(N);
    const uint LEVELS = prefix_levels.size();
//    std::cout << "array size: N = " << N << "\n";
//    std::cout << "count of buffers: LEVELS = " << LEVELS << "\n";
//
//    for(int i = 0; i < LEVELS; i++){
//        std::cout << "  size of level " << i << " = " << prefix_levels[i].size()/sizeof(uint) << "\n";
//    }


    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
//        std::cout << "Writing initial array... ";
        prefix_levels[0].writeN(arr.data(), N);
//        std::cout << "Ready.\n";

        t.restart();    // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

        for(int i = 0; i < LEVELS - 1; i++){
//            std::cout << "forward, step " << i << "\n";
//            print_gpu_array(prefix_levels[i]);

            uint size_i = prefix_levels[i].size() / sizeof(uint);
            uint size_i_plus_1 = prefix_levels[i+1].size() / sizeof(uint);
            auto ws = gpu::WorkSize(GROUP_SIZE/2, size_i_plus_1 * GROUP_SIZE/2);
            prefix_sum_forward.exec(ws, prefix_levels[i], prefix_levels[i + 1], size_i);

//            print_gpu_array(prefix_levels[i+1]);
        }

        for(int i = LEVELS - 2; i >= 0; i--){
//            std::cout << "backward, step " << i << "\n";
//            print_gpu_array(prefix_levels[i+1]);

            uint size_i = prefix_levels[i].size() / sizeof(uint);
            uint size_i_plus_1 = prefix_levels[i+1].size() / sizeof(uint);
            auto ws = gpu::WorkSize(GROUP_SIZE/2, size_i_plus_1 * GROUP_SIZE/2);
            prefix_sum_backward.exec(ws, prefix_levels[i], prefix_levels[i + 1], size_i);

//            print_gpu_array(prefix_levels[i]);
        }

        t.nextLap();
    }
    std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU: " << (N / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

    VEC<uint> res(N);
    prefix_levels[0].readN(res.data(), N);

    // Проверяем корректность результатов
    for (int i = 0; i < N; ++i) {
        EXPECT_THE_SAME(res[i], expected[i], "GPU results should be equal to CPU results!");
    }
}





int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

//    auto arr = std::vector<uint>(1024, 1);
//    auto expected = std::vector<uint>{1,2,3,4,5,6,7,8, 9};
//    test_gpu_prefix_sum(arr, expected, 1);
//
//    return 0;

	int benchmarkingIters = 10;
	unsigned int max_n = (1 << 24);
    unsigned int min_n = 4096;

	for (unsigned int n = min_n; n <= max_n; n *= 4) {
		std::cout << "______________________________________________" << std::endl;
		unsigned int values_range = std::min<unsigned int>(1023, std::numeric_limits<int>::max() / n);
		std::cout << "n=" << n << " values in range: [" << 0 << "; " << values_range << "]" << std::endl;

		std::vector<unsigned int> as(n, 0);
		FastRandom r(n);
		for (int i = 0; i < n; ++i) {
			as[i] = r.next(0, values_range);
		}

		std::vector<unsigned int> bs(n, 0);
		{
			for (int i = 0; i < n; ++i) {
				bs[i] = as[i];
				if (i) {
					bs[i] += bs[i-1];
				}
			}
		}
		const std::vector<unsigned int> reference_result = bs;

		{
			{
				std::vector<unsigned int> result(n);
				for (int i = 0; i < n; ++i) {
					result[i] = as[i];
					if (i) {
						result[i] += result[i-1];
					}
				}
				for (int i = 0; i < n; ++i) {
					EXPECT_THE_SAME(reference_result[i], result[i], "CPU result should be consistent!");
				}
			}

			std::vector<unsigned int> result(n);
			timer t;
			for (int iter = 0; iter < benchmarkingIters; ++iter) {
				for (int i = 0; i < n; ++i) {
					result[i] = as[i];
					if (i) {
						result[i] += result[i-1];
					}
				}
				t.nextLap();
			}
			std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
			std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
		}

        auto cs = as;
		{
            gpu::gpu_mem_32u cs_gpu;
            cs_gpu.resizeN(n);

            int log_n = 1;
            while((1 << log_n) < n)
                log_n += 1;

            auto work_size = gpu::WorkSize(128, n);
            ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");
            prefix_sum.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                cs_gpu.writeN(as.data(), n);
                t.restart();    // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

                for(uint step = 0; step < log_n; step++){
                    prefix_sum.exec(work_size, cs_gpu, step);
                }
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            cs_gpu.readN(cs.data(), n);
		}

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(cs[i], bs[i], "GPU results should be equal to CPU results!");
        }

        test_gpu_prefix_sum(as, bs, benchmarkingIters);
	}
}
