#include "cuda_Header.cuh"
#include "cuda_Error.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cpu_Header.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>


using namespace std;

//向量相加
__global__ void AddKernel(float* a, float* b, float* c, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int stride = blockIdx.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		c[i] = a[i] + b[i];
	}
}

// 矩阵类型，行优先，M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
	int width;//行宽
	int height;//列高
	float* elements;
};

// 获取矩阵A的(row, col)元素
__device__ float getElement(Matrix* A, int row, int col)
{
	return A->elements[row * A->width + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(Matrix* A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}

// 矩阵相乘kernel，2-D，每个线程计算一个元素
__global__ void matMulKernel(Matrix* A, Matrix* B, Matrix* C)
{
	float Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < A->width; ++i)
	{
		Cvalue += getElement(A, row, i) * getElement(B, i, col);
	}
	setElement(C, row, col, Cvalue);
}

//计算Box-Embedding相交的大小
__global__ void ComputeKernel(Matrix* R, Matrix* Q, Matrix* C, int dim, int N)
{
    double Cvalue = 1.0;//后续相乘，初始必须为1

    int strade = blockDim.x * gridDim.x;//跳步的大小，等于线程的宽

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-r,即行数，rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c,即列数, rowA=colC
    int rowA = colC;

    int step = dim / 2;
    //跳步循环，直到遍历所有数据N，则kernel对任意长度的N都可以实施
    while (colC < N)
    {        
        for (int d = 0; d < step; d++)
        {
            //计算每个维度A和B的box-embedding相交的大小
            double tmp = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))       //后半元素
                - max(getElement(R, rowA, d), getElement(Q, rowC, d));                           //前半元素
            //需要每个维度都大于0
            //if (tmp <= 0)//并行线程内的判断影响同步，尽量减少if分支
            //{
            //    Cvalue = -1.0;
            //    //break;
            //}
            //else
            //{   
            //Cvalue *= tmp;//小数多次相乘会导致数值接近0，由于精度原因数值消失vanishing，需做正则化Normalization
            //log(x)+log(y)=log(xy);log相加等于相乘的log，使得相乘计算的值得以保留；
            //但是log函数在[0,1]上的值域是负的，导致乘积结果不单调，若能保证tmp值小于1，则可保证Cvalue绝对值为单调的
            Cvalue += log(tmp);
            Cvalue = abs(Cvalue);
            //}
        }
        
        //Cvalues的z正则化Z-Normalization
        //均值
        //double mean = 0;
        //for (size_t i = 0; i < Cvalues.size(); i++)
        //{
        //    mean += Cvalues[i];
        //}
        //mean = mean / Cvalues.size();
        ////方差
        //double vari = 0;
        //for (size_t i = 0; i < Cvalues.size(); i++)
        //{
        //    vari += pow((Cvalues[i] - mean), 2);
        //}
        //vari = vari / Cvalues.size();

        setElement(C, rowC, colC, Cvalue);
        colC += strade;//跳步
    }
}

//线程足够时，无跳步循环
__global__ void ComputeKernel2(Matrix* R, Matrix* Q, Matrix* C, int dim)
{
    double Cvalue = 1.0;//后续相乘，初始必须为1

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-r,即行数，rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c,即列数, rowA=colC
    int rowA = colC;
    int step = dim / 2;
    
    for (int d = 0; d < step; d++)
    {
        //计算每个维度A和B的box-embedding相交的大小
        double tmp = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))       //后半元素
            - max(getElement(R, rowA, d), getElement(Q, rowC, d));                           //前半元素

        //需要每个维度都大于0,
        //Cvalue *= tmp;//小数多次相乘会导致数值接近0，数值消失vanishing，需做正则化Normalization
        // log正则化,且log之后，tmp小于零会得到nan
        Cvalue += log(tmp);

    }
    setElement(C, rowC, colC, Cvalue);
}

__global__ void SingleQueryKernel(Matrix* R, float* Q, float* C, int dim)
{
    float Cvalue = 1.0f;//后续相乘，初始必须为1

    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c,即列数, rowA=colC 64*32=2048
    int rowA = colC;
    int step = dim / 2;

    for (int d = 0; d < step; d++)
    {
        //计算每个维度A和B的box-embedding相交的大小
        float tmp = min(R->elements[rowA * R->width + d + step], Q[d + step])       //后半元素
            - max(R->elements[rowA * R->width + d], Q[d]);                          //前半元素

        //需要每个维度都大于0
        Cvalue *= exp(tmp);
    }
    C[colC]=Cvalue;
}

//获取GPU信息
void getCudaInformaton()
{
	int dev = 0;
	cudaDeviceProp devProp;
	CHECK(cudaGetDeviceProperties(&devProp, dev));
	std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
	std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
	std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
	std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

    int dev1 = 1;
    cudaDeviceProp devProp1;
    CHECK(cudaGetDeviceProperties(&devProp1, dev1));
    std::cout << "使用GPU device " << dev1 << ": " << devProp1.name << std::endl;
    std::cout << "SM的数量：" << devProp1.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp1.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp1.maxThreadsPerBlock << std::endl;
    std::cout << "每个SM的最大线程数：" << devProp1.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << devProp1.maxThreadsPerMultiProcessor / 32 << std::endl;
}

void CUDASingleQuery(char* argv[])
{
    //读数据
    Vars vars;
    vars.ReadPara(argv);
    vars.ReadOrgTextData();
    vars.ReadEmbData();
    vars.ShowPara();

    //以输出矩阵的高、宽为基准
    int height = 1;    //输出的高为查询的数量|q|=Qnums，单查询则为1，多个查询则为n
    int width = vars.EmbNums;   //输出的宽为记录的数量N，QNums若大于总线程数，则跳步循环
    int dim = vars.Dim;         //embedding维度D
    //矩阵Records[N*D],
    Matrix* R;
    //Query[1*D],Candidates[1*N]
    float* Q; float* C;

    // 申请托管内存初始化
    cudaMallocManaged((void**)&R, sizeof(Matrix));
    cudaMallocManaged((void**)&Q, sizeof(float));
    cudaMallocManaged((void**)&C, sizeof(float));

    //初始化数据R 
    R->width = dim;        //R的宽为embedding的维度D
    R->height = width;     //R的高为记录的数量N

    //申请托管内存
    int R_Bytes = R->width * R->height * sizeof(float);
    int Q_Bytes = dim * sizeof(float);
    int C_Bytes = width * sizeof(float);
    cudaMallocManaged((void**)&R->elements, R_Bytes);
    cudaMallocManaged((void**)&Q, Q_Bytes);
    cudaMallocManaged((void**)&C, C_Bytes);

    //取1个随机记录作为查询Q,若为jion问题，R自交，Q=R
    srand((unsigned)2024);
    int qid = rand() % vars.EmbNums;
    for (size_t j = 0; j < dim; j++)
    {
        Q[j] = vars.embeddings[qid][j];//qid行j列的值
        cout << j << ";" << Q[j] << std::endl;
    }

    //数据R加载
    for (size_t i = 0; i < R->height; i++)
    {
        for (size_t j = 0; j < R->width; j++)
        {
            R->elements[i * R->width + j] = vars.embeddings[i][j];//i行j列的值
        }
    }

    //一维的gird和block
    dim3 blockSize(64);
    dim3 gridSize((width+ blockSize.x -1)/blockSize.x);

    SingleQueryKernel << < gridSize, blockSize >> > (R, Q, C, dim);
    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();

    for (size_t i = 0; i < width; i++)
    {
        cout << i <<";" << C[i] << std::endl;
    }
}

void CUDACompute(char* argv[])
{
    //读数据
    Vars vars;
    vars.ReadPara(argv);    
    vars.ReadOrgTextData();
    vars.ReadEmbData();
    vars.ShowPara();
    
    clock_t start, end, start0;
    start0 = clock();
    start = clock();
    //以输出矩阵的高、宽为基准
    int height = vars.QNums;    //输出的高为查询的数量|q|=Qnums，单查询则为1，多个查询则为n
    int width = vars.EmbNums;   //输出的宽为记录的数量N，QNums若大于总线程数，则跳步循环
    int dim = vars.Dim;         //embedding维度D
    //矩阵Records[N*D],Query[|q|*D],Candidates[|q|*N]
    Matrix* R, * Q, * C;
    // 申请托管内存
    cudaMallocManaged((void**)&R, sizeof(Matrix));
    cudaMallocManaged((void**)&Q, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    
    // 初始化数据R 
    R->width = dim;        //R的宽为embedding的维度D
    R->height = width;     //R的高为记录的数量N
   
    // 初始化数据Q 
    Q->width = dim;         //Q的宽为embedding的维度D
    Q->height = height;     //Q的高为查询的数量|q|
    
    // 初始化数据C
    C->height = height;     //C的高为查询的数量|q|
    C->width = width;       //C的宽为记录的数量N

    //取QNums个随机记录作为查询Q,若为jion问题，R自交，Q=R
    srand((unsigned)2024);
    vector<int> queries;
    for (size_t n = 0; n < vars.QNums; n++)
    {
        int qid = rand() % vars.EmbNums;
        queries.push_back(qid);
        //cout << "QID = " << qid << endl;
    }

    //申请托管内存
    int R_Bytes = R->width * R->height * sizeof(float);
    int Q_Bytes = Q->width * Q->height * sizeof(float);
    int C_Bytes = C->width * C->height * sizeof(float);
    cudaMallocManaged((void**)&R->elements, R_Bytes);
    cudaMallocManaged((void**)&Q->elements, Q_Bytes);
    cudaMallocManaged((void**)&C->elements, C_Bytes);
    end = clock();
    cout << "GPUMemorytime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;

    start = clock();
    //数据加载，R，Q分开加载，Q在R加载的过程中加载会变慢！
    //数据R加载,数据加载必须在申请内存托管之后，否则无法加载到GPU
    for (size_t i = 0; i < R->height; i++)
    {
        for (size_t j = 0; j < R->width; j++)
        {
            R->elements[i * R->width + j] = vars.embeddings[i][j];//i行j列的值
        }
    }
    //数据Q加载
    sort(queries.begin(), queries.end());
    int q0id = 42;
    for (size_t i = 0; i < Q->height; i++)
    {
        int qid = queries[i];
        if (queries[i] == 1816)
            q0id = (int)i;
        for (size_t j = 0; j < Q->width; j++)
        {
            Q->elements[i * Q->width + j] = vars.embeddings[qid][j];//qid行j列的值
        }
    }
    end = clock();
    cout << "LoadDataToGPUtime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;

    start = clock();
    //SM的数量：10
    //每个线程块的共享内存大小：48 KB
    //每个线程块的最大线程数：1024
    //每个SM的最大线程数：2048
    //每个SM的最大线程束数：64
    //定义kernel的执行配置
    int blockx, blocky, gridx, gridy;
    blocky = 32;//高，设置为Qnum的最大公因子 
    blockx = 32;//宽，blockx*blocky是32的倍数，不小于64，不大于1024       
    gridy = (height + blocky - 1) / blocky;//高，固定为gridy*blocky=Qnum的长度，gridy<2^31-1  
    gridx = (width + blockx - 1) / blockx;//宽，gridx*gridy最好为sm的倍数2倍以上，且为32的倍数，实际数据不会总是32的倍数
    dim3 blockSize(blockx, blocky);//[32,32]
    dim3 gridSize(gridx, gridy);//[28, Qnums/32]

    //执行kernel//忽略错误提示
    ComputeKernel << < gridSize, blockSize >> > (R, Q, C, dim, width);
    
    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    
    end = clock();
    cout << "GPUComputetime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    cout << "GPUtime = " << double(end - start0) / CLOCKS_PER_SEC << "s" << endl;
    //for (size_t i = 0; i < C->width; i++)
    //{
    //    //C(row, col) = *(C.elements + row * C.width + col)
    //    if (C->elements[q0id*C->width + i]!=NULL)
    //    {
    //        std::cout << "ID：" << i << ";" << C->elements[q0id * C->width + i] << std::endl;
    //    }
    //    
    //}

    //save all candidates
    for (size_t i = 0; i < C->height; i++)
    {
        candDatas candidates;
        candidates.qID = queries[i];
        for (size_t j = 0; j < C->width; j++)
        {
            if (C->elements[i * C->width + j] > 0)
            {
                candData cd;
                cd.rID = (int)j;
                cd.boxIntersection = C->elements[i * C->width + j];
                candidates.cans.push_back(cd);
            }
        }
        vars.candidateSets.push_back(candidates);
    }

    for (size_t i = 0; i < queries.size(); i++)
    {        
        if (vars.SearchF == 3)
            vars.VerifyAllCandidates((int)i,queries[i]);
        if (vars.SearchF == 0)
            vars.topkSearch((int)i,queries[i]);
        if (vars.SearchF == 1)
            vars.rangeSearch((int)i,queries[i]);
    }
 
    //    ID：0; 3.13513
    //    ID：1; 0.123433
    //    ID：2; 0.64123
    //    ID：3; 0.0110187
    //    ID：4; 0.0161557
    //    ID：5; 8.66706e-05
    //    ID：6; 15461.4
    //    ID：16; 0.000491897
    //    ID：17; 0.000741085
    //    ID：24; 0.565358
    //    ID：26; 4.15953
    //    ID：27; 0.00755924
    // 
    //    ID：2401; 0.065052
    //    ID：2402; -1
    //    ID：2403; 0.116964
    //    ID：2404; 0.579599
    //    ID：2405; 0.00066764
    //    ID：2406; 0.0495014
    //    ID：2407; -1
    //    ID：2408; 0.0199137
    //    ID：2409; 6.76879
    //    ID：2410; -1
    //    ID：2411; -1
    //    ID：2412; 0.279201
    //    ID：2413; -1
    //    ID：2414; 0.00328183
    //    ID：2415; 0.00267098
    return;
}


void CUDA_main()
{	
    int width = 1024;
    int height = 1024;
    Matrix* A, * B, * C;
    // 申请托管内存
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void**)&A->elements, nBytes);
    cudaMallocManaged((void**)&B->elements, nBytes);
    cudaMallocManaged((void**)&C->elements, nBytes);

    // 初始化数据
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(32, 32);//32*32
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);//(1024+31)/32=32.97
    // 执行kernel//忽略错误提示
    matMulKernel << < gridSize, blockSize >> > (A, B, C);


    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    std::cout << "最大误差: " << maxError << std::endl;
    return;
}

