// cudaTest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include "cuda_Header.cuh"
#include "cpu_Header.h"
#include <iostream>
#include <time.h>
using namespace std;
int main(int argc, char* argv[])
{  
    clock_t start, end;

    //使用GPU device 0: NVIDIA GeForce GTX 1060 with Max - Q Design
    //SM的数量：10
    //每个线程块的共享内存大小：48 KB
    //每个线程块的最大线程数：1024
    //每个SM的最大线程数：2048
    //每个SM的最大线程束数：64
    //getCudaInformaton();

    //std::cout << "GPU计算:\n";    
    //start = clock();    
    //CUDACompute(argv); 
    ////CUDASingleQuery(argv);
    //end = clock();
    //cout << "time = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;   

    //std::cout << "CPU计算:\n";
    //start = clock();
    //CPUmain(argv);
    //end = clock();
    //cout << "CPU计算time = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;    

    return 0;

}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
