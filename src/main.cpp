#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


void aplusb_on_device(cl_platform_id platform, cl_device_id device, unsigned int n) {

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    cl_int errcode_ret;
    cl_context context = clCreateContext(0, 1, &device, nullptr, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);


    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);


    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    
    cl_mem buf_as = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*n, as.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    cl_mem buf_bs = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*n, bs.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    cl_mem buf_cs = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);


    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    cl_program program; 
    {
        const char* sources = kernel_sources.c_str();
        program = clCreateProgramWithSource(context, 1, &sources, 0, &errcode_ret);
        OCL_SAFE_CALL(errcode_ret);
    }

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo

    errcode_ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if(errcode_ret != 0){
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);

        if (log_size > 1) {
            std::cout << "Log:" << std::endl;
            std::cout << log.data() << std::endl << std::endl;
        }

        OCL_SAFE_CALL(errcode_ret);
    }

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_kernel kernel = clCreateKernel(program, "aplusb", &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &buf_as));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &buf_bs));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &buf_cs));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(unsigned int), &n));
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;    // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        cl_event event;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global_work_size, &workGroupSize, 0, 0, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();    // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "  Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "  GFlops: " << n / (t.lapAvg() * 1000 * 1000 * 1000) << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "  VRAM bandwidth: " << 3*n*sizeof(float) / ((1 << 30)*t.lapAvg()) << " GB/s" << std::endl;
        OCL_SAFE_CALL(clReleaseEvent(event));
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, buf_cs, CL_TRUE, 0, n * sizeof(float), cs.data(), 0, 0, 0));
            t.nextLap();
        }
        std::cout << "  Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "  VRAM -> RAM bandwidth: " << 1.0 * n * sizeof(float) / ((1 << 30) * t.lapAvg()) << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
       for (unsigned int i = 0; i < n; ++i) {
           if (cs[i] != as[i] + bs[i]) {
                std::cout << i << " " << cs[i] << std::endl;
                throw std::runtime_error("CPU and GPU results differ!");
           }
       }


    clReleaseKernel(kernel);

    clReleaseMemObject(buf_as);
    clReleaseMemObject(buf_bs);
    clReleaseMemObject(buf_cs);

    clReleaseProgram(program);

    clReleaseCommandQueue(queue);

    clReleaseContext(context);
}


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    unsigned int n = 100 * 1000 * 1000;
    std::cout << "Generate data for n=" << n << std::endl;
    std::cout << std::endl;

    // доступные платформы
    cl_uint platforms_count;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platforms_count));
    std::vector<cl_platform_id> platforms(platforms_count);
    OCL_SAFE_CALL(clGetPlatformIDs(platforms_count, platforms.data(), nullptr));

    for (cl_platform_id platform: platforms) {

        // доступные на платформе устройства
        cl_uint devices_count;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devices_count));
        std::vector<cl_device_id> devices(devices_count);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices_count, devices.data(), nullptr));

        for (cl_device_id device: devices) {
            // имя
            size_t device_name_size = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &device_name_size));
            std::vector<unsigned char> device_name(device_name_size);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_size, device_name.data(), nullptr));

            // тип
            cl_device_type device_type;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr));

            const char* device_type_name;
            if(device_type == CL_DEVICE_TYPE_CPU)       device_type_name = "CPU";
            else if(device_type == CL_DEVICE_TYPE_GPU)  device_type_name = "GPU";
            else                                        device_type_name = "not CPU and not GPU";

            // размер локальной памяти
            cl_ulong device_local_mem;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &device_local_mem, nullptr));
            device_local_mem /= (1 << 10);  // переводим в килобайты

            // размер глобальной памяти
            cl_ulong device_global_mem;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &device_global_mem, nullptr));
            device_global_mem /= (1 << 20);  // переводим в мегабайты

            // профиль: FULL_PROFILE - устройство аппаратно поддерживает OpenCL, EMBEDDED_PROFILE - если поддержка OpenCL программная
            size_t device_profile_size = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_PROFILE, 0, nullptr, &device_profile_size));
            std::vector<unsigned char> device_profile(device_profile_size);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_PROFILE, device_profile_size, device_profile.data(), nullptr));

            // версия OpenCL
            size_t device_opencl_version_size = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &device_opencl_version_size));    
            std::vector<unsigned char> device_opencl_version(device_opencl_version_size);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, device_opencl_version_size, device_opencl_version.data(), nullptr));

            std::cout << "Run aplusb on device: " \
                << device_name.data() << std::endl;
            std::cout << "Device properties: " \
                << device_type_name << ", " \
                << device_local_mem << "KB local mem, " \
                << device_global_mem << "MB global mem, " \
                << device_profile.data() << ", " \
                << device_opencl_version.data() << std::endl;
            std::cout << std::endl; 
            
            aplusb_on_device(platform, device, n);

            std::cout << std::endl;  
            std::cout << std::endl;         
        }
    }

    return 0;
}
