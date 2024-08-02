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

// M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
	int width;//row
	int height;//colume
	float* elements;
};

struct MultiMatrix
{
    int width;
    int height;
    float* ov;
    float* ja;
    float* cs;
    float* di;
};

bool simLesser(candData& a, candData& b)
{
    return a.sim < b.sim;
}

bool simLarger(candData& a, candData& b)
{
    return a.sim > b.sim;
}

bool resimLarger(resultData& a, resultData& b)
{
    return a.similarity > b.similarity;
}

//get element of A(row, col)
__device__ float getElement(Matrix* A, int row, int col)
{
	return A->elements[row * A->width + col];
}

__device__ float* getElement(MultiMatrix* A, int row, int col)
{
    float* values;
    values[0] = A->ov[row * A->width + col];
    values[1] = A->ja[row * A->width + col];
    values[2] = A->cs[row * A->width + col];
    values[3] = A->di[row * A->width + col];
    return values;
}

// set element of A(row, col)
__device__ void setElement(Matrix* A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}

__device__ void setElement(MultiMatrix* A, int row, int col, float value[])
{
    A->ov[row * A->width + col] = value[0];
    A->ja[row * A->width + col] = value[1];
    A->cs[row * A->width + col] = value[2];
    A->di[row * A->width + col] = value[3];
}

// matMul kernel，2-D
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

//Kernel computing all similarities
__global__ void ComputeAllSimsKernel(Matrix* R, Matrix* Q, MultiMatrix* C, int dim, int N)
{
    float Cvalue[4]{};
    //float EPS = std::numeric_limits<float>::min();
    int strade = blockDim.x * gridDim.x;//strade size

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-row，rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-col, rowA=colC
    int rowA = colC;

    int step = dim / 2;
    //loop for all data by strade, length=N
    while (colC < N)
    {
        float box_intersection = 0.0;
        float box_union = 0.0;
        float box_q = 0.0;
        float box_r = 0.0;
        for (int d = 0; d < step; d++)
        {
            //computer volumes of box-embeddings in each dimension
            float tmp_i = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))      
                - max(getElement(R, rowA, d), getElement(Q, rowC, d));                           

            float tmp_u = max(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))       
                - min(getElement(R, rowA, d), getElement(Q, rowC, d));                           

            float tmp_q = getElement(Q, rowC, d + step) - getElement(Q, rowC, d);
            float tmp_r = getElement(R, rowA, d + step) - getElement(R, rowA, d);
            //log regularization    
	    //if box_intersection<0, give it a small non-negtive value
            if (tmp_i <= 0)
                box_intersection = log(1e-10);
            else
                box_intersection += log(tmp_i);
            box_intersection += log(tmp_i);
            box_union += log(tmp_u);
            box_q += log(tmp_q);
            box_r += log(tmp_r);
            //}
        }
        Cvalue[0] = exp(box_intersection - max(box_q, box_r));                   //Overlpa
        Cvalue[1] = exp(box_intersection - box_union);                           //Jaccard
        Cvalue[2] = exp(box_intersection - (box_q + box_r) / 2);                 //Cosine
        Cvalue[3] = 2 * exp(box_intersection) / (exp(box_q) + exp(box_r) + 1e-10);//Dice
        setElement(C, rowC, colC, Cvalue);
        colC += strade;
    }
}

//Compute Overlap-C similarity
__global__ void ComputeOverlapCKernel(Matrix* R, Matrix* Q, Matrix* C, int dim, int N)
{
    double Cvalue = 0.0;

    int strade = blockDim.x * gridDim.x;

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-row，rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-col, rowA=colC
    int rowA = colC;

    int step = dim / 2;
    //loop for all data by strade, length=N
    while (colC < N)
    {        
        float box_intersection = 0.0;
        float box_q = 0.0;
        float box_r = 0.0;
        for (int d = 0; d < step; d++)
        {
             //computer volumes of box-embeddings in each dimension
            float tmp_i = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))       
                - max(getElement(R, rowA, d), getElement(Q, rowC, d));                          
            float tmp_q = getElement(Q, rowC, d + step) - getElement(Q, rowC, d);
            float tmp_r = getElement(R, rowA, d + step) - getElement(R, rowA, d);
           
            //log regularization    
	    //if box_intersection<0, give it a small non-negtive EPS value
            if (tmp_i <= 0)
            {
                box_intersection = log(1e-10);
                box_r=box_q = 0;                
                break;
            }
            else
                box_intersection += log(tmp_i);
            box_q += log(tmp_q);
            box_r += log(tmp_r);
        }
        Cvalue = exp(box_intersection - max(box_q, box_r));
        setElement(C, rowC, colC, Cvalue);
        colC += strade;
    }
}

//Compute Jaccard similarity
__global__ void ComputeJaccardKernel(Matrix* R, Matrix* Q, Matrix* C, int dim, int N)
{
    double Cvalue = 0.0;
    
    int strade = blockDim.x * gridDim.x;

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-r,即行数，rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c,即列数, rowA=colC
    int rowA = colC;

    int step = dim / 2;
    
    while (colC < N)
    {        
        float box_intersection = 0.0;
        float box_union = 0.0;
        for (int d = 0; d < step; d++)
        {           
            float tmp_i = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))       
                - max(getElement(R, rowA, d), getElement(Q, rowC, d));                           

            float tmp_u = max(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))       
                - min(getElement(R, rowA, d), getElement(Q, rowC, d));                          

            if (tmp_i <= 0)
            {
                box_intersection = log(1e-10);
                box_union = 0;                
                break;
            }
            else
                box_intersection += log(tmp_i);
            box_union += log(tmp_u);
        }
        Cvalue = exp(box_intersection- box_union);
        setElement(C, rowC, colC, Cvalue);
        colC += strade;
    }
}

//Compute Cosine similarity
__global__ void ComputeCosineKernel(Matrix* R, Matrix* Q, Matrix* C, int dim, int N)
{
    double Cvalue = 0.0;

    int strade = blockDim.x * gridDim.x;

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-r,rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c,rowA=colC
    int rowA = colC;

    int step = dim / 2;
    
    while (colC < N)
    {        
        float box_intersection = 0.0;
        float box_q = 0.0;
        float box_r = 0.0;
        for (int d = 0; d < step; d++)
        {
            
            float tmp_i = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))      
                - max(getElement(R, rowA, d), getElement(Q, rowC, d));                           
            float tmp_q = getElement(Q, rowC, d + step) - getElement(Q, rowC, d);
            float tmp_r = getElement(R, rowA, d + step) - getElement(R, rowA, d);
            
            if (tmp_i <= 0)
            {
                box_intersection = log(1e-10);
                box_r = box_q = 0;
                break;
            }
            else
                box_intersection += log(tmp_i);
            box_q += log(tmp_q);
            box_r += log(tmp_r);
        }
        Cvalue = exp(box_intersection - (box_q+ box_r)/2);
        setElement(C, rowC, colC, Cvalue);
        colC += strade;
    }
}

//Compute Dice similarity
__global__ void ComputeDiceKernel(Matrix* R, Matrix* Q, Matrix* C, int dim, int N)
{
    double Cvalue = 0.0;

    int strade = blockDim.x * gridDim.x;

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-r,rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c,rowA=colC
    int rowA = colC;

    int step = dim / 2;
    
    while (colC < N)
    {
        float box_intersection = 0.0;
        float box_q = 0.0;
        float box_r = 0.0;
        for (int d = 0; d < step; d++)
        {
            
            float tmp_i = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))      
                - max(getElement(R, rowA, d), getElement(Q, rowC, d));                           
            float tmp_q = getElement(Q, rowC, d + step) - getElement(Q, rowC, d);
            float tmp_r = getElement(R, rowA, d + step) - getElement(R, rowA, d);
            
            if (tmp_i <= 0)
            {
                box_intersection = log(1e-10);
                box_r = box_q = 0;
                break;
            }
            else
                box_intersection += log(tmp_i);
            box_q += log(tmp_q);
            box_r += log(tmp_r);
        }
        Cvalue = 2*exp(box_intersection)/ (exp(box_q) + exp(box_r) + 1e-10);
        setElement(C, rowC, colC, Cvalue);
        colC += strade;
    }
}

//compute Box-intersection
__global__ void ComputeKernel(Matrix* R, Matrix* Q, Matrix* C, int dim, int N)
{
    double Cvalue = 0.0;

    int strade = blockDim.x * gridDim.x;

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-r,rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c, rowA=colC
    int rowA = colC;

    int step = dim / 2;
    
    while (colC < N)
    {        
        for (int d = 0; d < step; d++)
        {
            
            double tmp = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))       
                - max(getElement(R, rowA, d), getElement(Q, rowC, d));                           

            if (tmp <= 0)
                Cvalue = log(1e-10);
            else
                Cvalue += log(tmp);
                       
        }
        Cvalue = exp(Cvalue);
        setElement(C, rowC, colC, Cvalue);
        colC += strade;
    }
}

//no strade
__global__ void ComputeKernel2(Matrix* R, Matrix* Q, Matrix* C, int dim)
{
    double Cvalue = 0.0;

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-r, rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c, rowA=colC
    int rowA = colC;
    int step = dim / 2;
    
    for (int d = 0; d < step; d++)
    {
        
        double tmp = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))       
            - max(getElement(R, rowA, d), getElement(Q, rowC, d));                           

       if (tmp <= 0)
            tmp = 1e-6;
        Cvalue += log(tmp);

    }
    Cvalue = exp(Cvalue);
    setElement(C, rowC, colC, Cvalue);
}

__global__ void SingleQueryKernel(Matrix* R, float* Q, float* C, int dim)
{
    float Cvalue = 0.0;

    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c, rowA=colC
    int rowA = colC;
    int step = dim / 2;

    for (int d = 0; d < step; d++)
    {
        
        float tmp = min(R->elements[rowA * R->width + d + step], Q[d + step])      
            - max(R->elements[rowA * R->width + d], Q[d]);                         
        
        if (tmp <= 0)
            tmp = 1e-6;
        Cvalue += log(tmp);
    }
    Cvalue = exp(Cvalue);
    C[colC]=Cvalue;
}

//get GPU information
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

    //int dev1 = 1;
    //cudaDeviceProp devProp1;
    //CHECK(cudaGetDeviceProperties(&devProp1, dev1));
    //std::cout << "使用GPU device " << dev1 << ": " << devProp1.name << std::endl;
    //std::cout << "SM的数量：" << devProp1.multiProcessorCount << std::endl;
    //std::cout << "每个线程块的共享内存大小：" << devProp1.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    //std::cout << "每个线程块的最大线程数：" << devProp1.maxThreadsPerBlock << std::endl;
    //std::cout << "每个SM的最大线程数：" << devProp1.maxThreadsPerMultiProcessor << std::endl;
    //std::cout << "每个SM的最大线程束数：" << devProp1.maxThreadsPerMultiProcessor / 32 << std::endl;
}

void CUDASingleQuery(char* argv[])
{
    //读数据
    Vars vars;
    bool para = vars.ReadPara(argv);
    if (!para)
    {
        cout << "parameter error" << endl;
        return;
    }
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
    getCudaInformaton();
    //read data
    Vars vars;
    bool para = vars.ReadPara(argv);
    if (!para)
    {
        cout << "parameter error" << endl;
        return;
    }
    vars.ReadOrgTextData();
    vars.ReadEmbData();
    vars.ShowPara();
    
    clock_t start, end, start0, end0;
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
    //数据加载必须在申请内存托管之后，否则无法加载到GPU
    for (size_t i = 0; i < R->height; i++)
    {
        for (size_t j = 0; j < R->width; j++)
        {
            R->elements[i * R->width + j] = vars.embeddings[i][j];//i行j列的值
        }
    }
    //数据Q加载
    sort(queries.begin(), queries.end());
    //int q0id = 42;
    for (size_t i = 0; i < Q->height; i++)
    {
        int qid = queries[i];
        //if (queries[i] == 1816)
        //    q0id = (int)i;
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
    dim3 gridSize(gridx, gridy);//

    //kernel
    //ComputeKernel << < gridSize, blockSize >> > (R, Q, C, dim, width);

    if ((SimilarityF)vars.SimF == Overlap)
        ComputeOverlapCKernel << < gridSize, blockSize >> > (R, Q, C, dim, width);
    else if ((SimilarityF)vars.SimF == Jaccard)
        ComputeJaccardKernel << < gridSize, blockSize >> > (R, Q, C, dim, width);
    else if ((SimilarityF)vars.SimF == Cosine)
        ComputeCosineKernel << < gridSize, blockSize >> > (R, Q, C, dim, width);
    else if ((SimilarityF)vars.SimF == Dice)
        ComputeDiceKernel << < gridSize, blockSize >> > (R, Q, C, dim, width);
    else
        return;
    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    
    end0 = end = clock();
    cout << "GPUSearchTime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    
    //Save Cadidates
    start = clock();
    for (size_t i = 0; i < C->height; i++)
    {
        candDatas candidates;
        candidates.qID = queries[i];
        for (size_t j = 0; j < C->width; j++)
        {
            if (C->elements[i * C->width + j] > 1e-6)
            {
                candData cd;
                cd.rID = (int)j;
                cd.sim = C->elements[i * C->width + j];
                candidates.cans.push_back(cd);
            }

        }
        vars.candidateSets.push_back(candidates);
        if (vars.candidateSets[i].cans.size() > 0)
            sort(vars.candidateSets[i].cans.begin(), vars.candidateSets[i].cans.end(), simLarger);
    }
    end = clock();
    cout << "GPUSaveCandsTime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    //Verfication phase
    start = clock();
    for (size_t i = 0; i < queries.size(); i++)
    {
        if (vars.SearchF == 0)
            vars.topkSearch((int)i, queries[i]);
        if (vars.SearchF == 1)
            vars.rangeSearch((int)i, queries[i]);
    }
    end = clock();
    cout << "VerifyTime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    cout << "totalGPUTime = " << double(end - start0) / CLOCKS_PER_SEC << "s" << endl;
    //Verfication method 2
    start = clock();
    //check values from GPU and save candidates whose values rank in top lambdaK*k
    if (vars.SearchF==0)
    {
        size_t sizeK = (size_t)ceil(vars.k * vars.lambdaK);        
        for (size_t i = 0; i < C->height; i++)
        {
            //save candidates, can be skipped
            //candDatas candidates;
            //candidates.qID = queries[i];
            //for (size_t j = 0; j < C->width; j++)
            //{
            //    float boxSim = C->elements[i * C->width + j];
            //    if (boxSim > 1e-6)
            //    {
            //        if (candidates.cans.size() < sizeK)
            //        {
            //            candData cd;
            //            cd.rID = (int)j;
            //            cd.sim = boxSim;
            //            candidates.cans.push_back(cd);
            //        }
            //        else
            //        {
            //            sort(candidates.cans.begin(), candidates.cans.end(), simLarger);
            //            if (boxSim > candidates.cans[sizeK].sim)
            //            {
            //                candidates.cans[sizeK - 1].rID = j;
            //                candidates.cans[sizeK - 1].sim = boxSim;
            //            }
            //        }
            //    }
            //}
            //vars.candidateSets.push_back(candidates);

            //check results
            int qid = queries[i];
            resultDatas tmpresdatas;
            tmpresdatas.qID = qid;

            resultDatas resdatas;
            resdatas.qID = qid;

            for (size_t j = 0; j < C->width; j++)
            {
                float boxSim = C->elements[i * C->width + j];
                if (boxSim > 1e-6)
                {
                    float maxSim = -1.0f;
                    int count = 0;
                    if (tmpresdatas.res.size() < sizeK)
                    {
                        float similarity = vars.Similarity(vars.SimF, vars.orgText[queries[i]].record, vars.orgText[(int)j].record);
                        resultData rd;
                        rd.rID = (int)j;
                        rd.similarity = similarity;
                        tmpresdatas.res.push_back(rd);
                        if (tmpresdatas.res.size() == sizeK)
                        {
                            sort(tmpresdatas.res.begin(), tmpresdatas.res.end(), resimLarger);
                            maxSim = tmpresdatas.res[sizeK - 1].similarity;
                            for (size_t c = 0; c < vars.k; c++)
                            {
                                resdatas.res.push_back(tmpresdatas.res[c]);
                            }
                        }
                    }
                    else
                    {
                        if (boxSim >= maxSim)
                        {
                            float similarity = vars.Similarity(vars.SimF, vars.orgText[queries[i]].record, vars.orgText[(int)j].record);
                            if (similarity >= resdatas.res[vars.k - 1].similarity)
                            {                                
                                resdatas.res[vars.k - 1].rID = (int)j;
                                resdatas.res[vars.k - 1].similarity = similarity;                                
                                sort(resdatas.res.begin(), resdatas.res.end(), resimLarger);
                                count++;
                                maxSim = tmpresdatas.res[sizeK - 1 - count].similarity;
                            }
                        }
                    }

                }

            }
            vars.resultSets.push_back(resdatas);
        }

    }
    //check values from GPU and save candidates whose values are in range of lambdaRange*range
    else if (vars.SearchF == 1)
    {        
        float tmpRange = vars.range/ vars.lambdaRange;
        for (size_t i = 0; i < C->height; i++)
        {
            //save candidates, can be skipped
            //int qid = queries[i];
            //candDatas candidates;
            //candidates.qID = qid;
            //for (size_t j = 0; j < C->width; j++)
            //{
            //    float boxSim = C->elements[i * C->width + j];
            //    if (boxSim > tmpRange)
            //    {
            //        candData cd;
            //        cd.rID = (int)j;
            //        cd.sim = boxSim;
            //        candidates.cans.push_back(cd);
            //    }
            //}
            //vars.candidateSets.push_back(candidates);

            //check results
            int qid = queries[i];
            resultDatas resdatas;
            resdatas.qID = qid;
            for (size_t j = 0; j < C->width; j++)
            {
                float boxSim = C->elements[i * C->width + j];
                if (boxSim > tmpRange)
                {
                    resultData rd;
                    rd.rID = (int)j;
                    rd.similarity = vars.Similarity(vars.SimF, vars.orgText[queries[i]].record, vars.orgText[(int)j].record);
                    if (rd.similarity >= vars.range)
                        resdatas.res.push_back(rd);
                }
            }
            vars.resultSets.push_back(resdatas);
        }
    }    

    end = clock();
    cout << "VerifyTime2 = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl; 
    cout << "totalGPUTime2 = " << double(end - start + end0 - start0) / CLOCKS_PER_SEC << "s" << endl;
    return;
}

// Compute all similarities on device
void CUDAComputeAllSims(char* argv[])
{
    //读数据
    Vars vars;
    bool para = vars.ReadPara(argv);
    if (!para)
    {
        cout << "parameter error = " << endl;
        return;
    }
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
    Matrix* R, * Q;
    MultiMatrix* C;
    // 申请托管内存
    cudaMallocManaged((void**)&R, sizeof(Matrix));
    cudaMallocManaged((void**)&Q, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(MultiMatrix));

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
    cudaMallocManaged((void**)&C->ov, C_Bytes);
    cudaMallocManaged((void**)&C->ja, C_Bytes);
    cudaMallocManaged((void**)&C->cs, C_Bytes);
    cudaMallocManaged((void**)&C->di, C_Bytes);
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
    ComputeAllSimsKernel << < gridSize, blockSize >> > (R, Q, C, dim, width);

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();

    end = clock();
    cout << "GPUComputetime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    cout << "GPUtime = " << double(end - start0) / CLOCKS_PER_SEC << "s" << endl;

    //save all candidates
    for (size_t i = 0; i < C->height; i++)
    {
        candMultiDatas candidates;
        candidates.qID = queries[i];
        for (size_t j = 0; j < C->width; j++)
        {
            
            candMultiData cd;
            cd.rID = (int)j;
            cd.ov = C->ov[i * C->width + j];
            cd.ja = C->ja[i * C->width + j];
            cd.cs = C->cs[i * C->width + j];
            cd.di = C->di[i * C->width + j];
            candidates.cans.push_back(cd);
            
        }
        vars.candiMultidateSets.push_back(candidates);
    }

    for (size_t i = 0; i < queries.size(); i++)
    {
        if (vars.SearchF == 3)
            vars.VerifyAllCandidates((int)i, queries[i]);
        if (vars.SearchF == 0)
            vars.topkSearch((int)i, queries[i]);
        if (vars.SearchF == 1)
            vars.rangeSearch((int)i, queries[i]);
    }
    return;
}

