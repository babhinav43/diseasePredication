#include <iostream>
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

constexpr size_t N = 1024;

void matrixAdd(sycl::queue& queue, float* A, float* B, float* C) {
    queue.submit([&](sycl::handler& cgh) {
        sycl::range<2> globalSize(N, N);
        sycl::range<2> localSize(16, 16);

        cgh.parallel_for(sycl::nd_range<2>(globalSize, localSize), [=](sycl::nd_item<2> item) {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);
            size_t idx = i * N + j;
            C[idx] = A[idx] + B[idx];
        });
    });
    queue.wait();
}

int main() {
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);

    try {
        sycl::queue queue(sycl::gpu_selector{});

        sycl::buffer<float, 2> bufferA(A.data(), sycl::range<2>(N, N));
        sycl::buffer<float, 2> bufferB(B.data(), sycl::range<2>(N, N));
        sycl::buffer<float, 2> bufferC(C.data(), sycl::range<2>(N, N));

        matrixAdd(queue, bufferA.get_access<sycl::access::mode::read>(queue),
                  bufferB.get_access<sycl::access::mode::read>(queue),
                  bufferC.get_access<sycl::access::mode::write>(queue));

        // Retrieve the result from the buffer
        sycl::host_accessor result(bufferC, sycl::read_only);
        std::cout << "Result: " << result[0][0] << std::endl;
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
