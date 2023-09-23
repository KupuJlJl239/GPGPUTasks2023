cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(task3)

set(CMAKE_CXX_STANDARD 11)

# OpenMP позволит распараллеливать циклы на все ядра процессора простыми директивами

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found!")
endif()

# convertIntoHeader CMake-функция объявлена в libs/gpu/CMakeLists.txt:71
# Она считывает все байты из файла src/cl/aplusb.cl (т.е. весь исходный код кернела) и преобразует их в массив байтов в файле src/cl/aplusb_cl.h aplusb_kernel
# Обратите внимание что это происходит на этапе компиляции, кроме того необходимо чтобы файл src/cl/aplusb_cl.h был перечислен среди исходников для компиляции при вызове add_executable
convertIntoHeader(src/cl/aplusb.cl src/cl/h/aplusb_cl.h aplusb_kernel)
add_executable(aplusb src/main_aplusb.cpp src/cl/h/aplusb_cl.h)
target_link_libraries(aplusb libclew libgpu libutils)

# Аналогично для фрактала Мандельброта
convertIntoHeader(src/cl/mandelbrot.cl src/cl/h/mandelbrot_cl.h mandelbrot_kernel)
add_executable(mandelbrot src/main_mandelbrot.cpp src/cl/h/mandelbrot_cl.h)
target_link_libraries(mandelbrot libclew libgpu libimages libutils)

# Для вычисления суммы
convertIntoHeader(src/cl/sum_base.cl src/cl/h/sum_base_cl.h sum_base_kernel)
convertIntoHeader(src/cl/sum_cycle.cl src/cl/h/sum_cycle_cl.h sum_cycle_kernel)
convertIntoHeader(src/cl/sum_coalesced.cl src/cl/h/sum_coalesced_cl.h sum_coalesced_kernel)
convertIntoHeader(src/cl/sum_local_mem.cl src/cl/h/sum_local_mem_cl.h sum_local_mem_kernel)
convertIntoHeader(src/cl/sum_tree.cl src/cl/h/sum_tree_cl.h sum_tree_kernel)

add_executable(sum src/main_sum.cpp
        src/cl/h/sum_base_cl.h
        src/cl/h/sum_cycle_cl.h
        src/cl/h/sum_coalesced_cl.h
        src/cl/h/sum_local_mem_cl.h
        src/cl/h/sum_tree_cl.h
)
target_link_libraries(sum libclew libgpu libutils)
