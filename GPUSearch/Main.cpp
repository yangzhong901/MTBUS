#include "cuda_Header.cuh"
#include "cpu_Header.h"
#include <iostream>
#include <time.h>
using namespace std;
int main(int argc, char* argv[])
{  
    std::clock_t start, end;

    if (atoi(argv[8])== 0)
    {
        std::cout << "GPU Running:\n";
        start = std::clock();
        CUDACompute(argv);
        end = std::clock();
        cout << "CUDAtime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    }
    if (atoi(argv[8]) == 1)
    {
        std::cout << "CPU Running:\n";
        start = clock();
        CPUmain(argv);
        end = clock();
        cout << "CPUtime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    }

    return 0;

}
