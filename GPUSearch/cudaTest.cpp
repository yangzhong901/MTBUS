#include "cuda_Header.cuh"
#include "cpu_Header.h"
#include <iostream>
#include <time.h>
using namespace std;
int main(int argc, char* argv[])
{  
    std::clock_t start, end;

    //使用GPU device 0: 
    //SM的数量：10
    //每个线程块的共享内存大小：48 KB
    //每个线程块的最大线程数：1024
    //每个SM的最大线程数：2048
    //每个SM的最大线程束数：64
    
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
