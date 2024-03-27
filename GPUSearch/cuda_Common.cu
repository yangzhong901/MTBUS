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

//�������
__global__ void AddKernel(float* a, float* b, float* c, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int stride = blockIdx.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		c[i] = a[i] + b[i];
	}
}

// �������ͣ������ȣ�M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
	int width;//�п�
	int height;//�и�
	float* elements;
};

// ��ȡ����A��(row, col)Ԫ��
__device__ float getElement(Matrix* A, int row, int col)
{
	return A->elements[row * A->width + col];
}

// Ϊ����A��(row, col)Ԫ�ظ�ֵ
__device__ void setElement(Matrix* A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}

// �������kernel��2-D��ÿ���̼߳���һ��Ԫ��
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

//����Box-Embedding�ཻ�Ĵ�С
__global__ void ComputeKernel(Matrix* R, Matrix* Q, Matrix* C, int dim, int N)
{
    float Cvalue = 1.0;//������ˣ���ʼ����Ϊ1

    int strade = blockDim.x * gridDim.x;//�����Ĵ�С�������̵߳Ŀ�

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-r,��������rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c,������, rowA=colC
    int rowA = colC;

    int step = dim / 2;
    //����ѭ����ֱ��������������N����kernel�����ⳤ�ȵ�N������ʵʩ
    while (colC < N)
    {        
        for (int d = 0; d < step; d++)
        {
            //����ÿ��ά��A��B��box-embedding�ཻ�Ĵ�С
            double tmp = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))       //���Ԫ��
                - max(getElement(R, rowA, d), getElement(Q, rowC, d));                           //ǰ��Ԫ��
            //��Ҫÿ��ά�ȶ�����0
            if (tmp <= 0)//�����߳��ڵ��ж�Ӱ��ͬ������������if��֧
            {
                Cvalue = -1.0;
                //break;
            }
            else
            {    ////Cvalue *= tmp;//С�������˻ᵼ����ֵ�ӽ�0����ֵ��ʧvanishing����������Normalization
                // log���򻯣�����ӣ���ȡ����
                Cvalue +=tmp;
            }
        }
        Cvalue = (float)abs(log10(Cvalue));
        //Cvalues��z����Z-Normalization
        //��ֵ
        //double mean = 0;
        //for (size_t i = 0; i < Cvalues.size(); i++)
        //{
        //    mean += Cvalues[i];
        //}
        //mean = mean / Cvalues.size();
        ////����
        //double vari = 0;
        //for (size_t i = 0; i < Cvalues.size(); i++)
        //{
        //    vari += pow((Cvalues[i] - mean), 2);
        //}
        //vari = vari / Cvalues.size();

        setElement(C, rowC, colC, Cvalue);
        colC += strade;//����
    }
}

//�߳��㹻ʱ��������ѭ��
__global__ void ComputeKernel2(Matrix* R, Matrix* Q, Matrix* C, int dim)
{
    double Cvalue = 1.0;//������ˣ���ʼ����Ϊ1

    int rowC = threadIdx.y + blockIdx.y * blockDim.y; //blockId-r,��������rowB=rowC
    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c,������, rowA=colC
    int rowA = colC;
    int step = dim / 2;
    
    for (int d = 0; d < step; d++)
    {
        //����ÿ��ά��A��B��box-embedding�ཻ�Ĵ�С
        double tmp = min(getElement(R, rowA, d + step), getElement(Q, rowC, d + step))       //���Ԫ��
            - max(getElement(R, rowA, d), getElement(Q, rowC, d));                           //ǰ��Ԫ��

        //��Ҫÿ��ά�ȶ�����0,
        //Cvalue *= tmp;//С�������˻ᵼ����ֵ�ӽ�0����ֵ��ʧvanishing����������Normalization
        // log����,��log֮��tmpС�����õ�nan
        Cvalue *= abs(log10(tmp));

    }
    setElement(C, rowC, colC, Cvalue);
}

__global__ void SingleQueryKernel(Matrix* R, float* Q, float* C, int dim)
{
    float Cvalue = 1.0f;//������ˣ���ʼ����Ϊ1

    int colC = threadIdx.x + blockIdx.x * blockDim.x; //blockId-c,������, rowA=colC 64*32=2048
    int rowA = colC;
    int step = dim / 2;

    for (int d = 0; d < step; d++)
    {
        //����ÿ��ά��A��B��box-embedding�ཻ�Ĵ�С
        float tmp = min(R->elements[rowA * R->width + d + step], Q[d + step])       //���Ԫ��
            - max(R->elements[rowA * R->width + d], Q[d]);                          //ǰ��Ԫ��

        //��Ҫÿ��ά�ȶ�����0
        Cvalue *= abs(log10(tmp));
        //if (tmp < 0)//�����߳��ڵ��ж�Ӱ��ͬ������������if��֧
        //{
        //    Cvalue = -1.0f;
        //    //break;
        //}
        //else if (tmp==0)
        //{
        //    Cvalue = 0;
        //}
        //else
        //{   //log����
        //    Cvalue *= abs(log10(tmp));
        //}
    }
    C[colC]=Cvalue;
}

//��ȡGPU��Ϣ
void getCudaInformaton()
{
	int dev = 0;
	cudaDeviceProp devProp;
	CHECK(cudaGetDeviceProperties(&devProp, dev));
	std::cout << "ʹ��GPU device " << dev << ": " << devProp.name << std::endl;
	std::cout << "SM��������" << devProp.multiProcessorCount << std::endl;
	std::cout << "ÿ���߳̿�Ĺ����ڴ��С��" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	std::cout << "ÿ���߳̿������߳�����" << devProp.maxThreadsPerBlock << std::endl;
	std::cout << "ÿ��SM������߳�����" << devProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "ÿ��SM������߳�������" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}

void CUDASingleQuery(char* argv[])
{
    //������
    Vars vars;
    vars.ReadPara(argv);
    vars.ReadOrgTextData();
    vars.ReadEmbData();
    vars.ShowPara();

    //���������ĸߡ���Ϊ��׼
    int height = 1;    //����ĸ�Ϊ��ѯ������|q|=Qnums������ѯ��Ϊ1�������ѯ��Ϊn
    int width = vars.EmbNums;   //����Ŀ�Ϊ��¼������N��QNums���������߳�����������ѭ��
    int dim = vars.Dim;         //embeddingά��D
    //����Records[N*D],
    Matrix* R;
    //Query[1*D],Candidates[1*N]
    float* Q; float* C;

    // �����й��ڴ��ʼ��
    cudaMallocManaged((void**)&R, sizeof(Matrix));
    cudaMallocManaged((void**)&Q, sizeof(float));
    cudaMallocManaged((void**)&C, sizeof(float));

    // ��ʼ������R 
    R->width = dim;        //R�Ŀ�Ϊembedding��ά��D
    R->height = width;     //R�ĸ�Ϊ��¼������N

    //�����й��ڴ�
    int R_Bytes = R->width * R->height * sizeof(float);
    int Q_Bytes = dim * sizeof(float);
    int C_Bytes = width * sizeof(float);
    cudaMallocManaged((void**)&R->elements, R_Bytes);
    cudaMallocManaged((void**)&Q, Q_Bytes);
    cudaMallocManaged((void**)&C, C_Bytes);

    //ȡ1�������¼��Ϊ��ѯQ,��Ϊjion���⣬R�Խ���Q=R
    srand((unsigned)2024);
    int qid = rand() % vars.EmbNums;
    for (size_t j = 0; j < dim; j++)
    {
        Q[j] = vars.embeddings[qid][j];//qid��j�е�ֵ
        cout << j << ";" << Q[j] << std::endl;
    }

    //����R����
    for (size_t i = 0; i < R->height; i++)
    {
        for (size_t j = 0; j < R->width; j++)
        {
            R->elements[i * R->width + j] = vars.embeddings[i][j];//i��j�е�ֵ
        }
    }

    //һά��gird��block
    dim3 blockSize(64);
    dim3 gridSize((width+ blockSize.x -1)/blockSize.x);

    SingleQueryKernel << < gridSize, blockSize >> > (R, Q, C, dim);
    // ͬ��device ��֤�������ȷ����
    cudaDeviceSynchronize();

    for (size_t i = 0; i < width; i++)
    {
        cout << i <<";" << C[i] << std::endl;
    }
}

void CUDACompute(char* argv[])
{
    //������
    Vars vars;
    vars.ReadPara(argv);    
    vars.ReadOrgTextData();
    vars.ReadEmbData();
    vars.ShowPara();
    
    //���������ĸߡ���Ϊ��׼
    int height = vars.QNums;    //����ĸ�Ϊ��ѯ������|q|=Qnums������ѯ��Ϊ1�������ѯ��Ϊn
    int width = vars.EmbNums;   //����Ŀ�Ϊ��¼������N��QNums���������߳�����������ѭ��
    int dim = vars.Dim;         //embeddingά��D
    //����Records[N*D],Query[|q|*D],Candidates[|q|*N]
    Matrix* R, * Q, * C;
    // �����й��ڴ�
    cudaMallocManaged((void**)&R, sizeof(Matrix));
    cudaMallocManaged((void**)&Q, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    
    // ��ʼ������R 
    R->width = dim;        //R�Ŀ�Ϊembedding��ά��D
    R->height = width;     //R�ĸ�Ϊ��¼������N
   
    // ��ʼ������Q 
    Q->width = dim;         //Q�Ŀ�Ϊembedding��ά��D
    Q->height = height;     //Q�ĸ�Ϊ��ѯ������|q|
    
    // ��ʼ������C
    C->height = height;     //C�ĸ�Ϊ��ѯ������|q|
    C->width = width;       //C�Ŀ�Ϊ��¼������N

    //ȡQNums�������¼��Ϊ��ѯQ,��Ϊjion���⣬R�Խ���Q=R
    srand((unsigned)2024);
    vector<int> queries;
    for (size_t n = 0; n < vars.QNums; n++)
    {
        int qid = rand() % vars.EmbNums;
        queries.push_back(qid);
        //cout << "QID = " << qid << endl;
    }

    //�����й��ڴ�
    int R_Bytes = R->width * R->height * sizeof(float);
    int Q_Bytes = Q->width * Q->height * sizeof(float);
    int C_Bytes = C->width * C->height * sizeof(float);
    cudaMallocManaged((void**)&R->elements, R_Bytes);
    cudaMallocManaged((void**)&Q->elements, Q_Bytes);
    cudaMallocManaged((void**)&C->elements, C_Bytes);

    clock_t start, end;
    start = clock();
    //���ݼ��أ�R��Q�ֿ����أ�Q��R���صĹ����м��ػ������
    //����R����,���ݼ��ر����������ڴ��й�֮�󣬷����޷����ص�GPU
    for (size_t i = 0; i < R->height; i++)
    {
        for (size_t j = 0; j < R->width; j++)
        {
            R->elements[i * R->width + j] = vars.embeddings[i][j];//i��j�е�ֵ
        }
    }
    //����Q����
    sort(queries.begin(), queries.end());
    int q0id = 42;
    for (size_t i = 0; i < Q->height; i++)
    {
        int qid = queries[i];
        if (queries[i] == 1816)
            q0id = (int)i;
        for (size_t j = 0; j < Q->width; j++)
        {
            Q->elements[i * Q->width + j] = vars.embeddings[qid][j];//qid��j�е�ֵ
        }
    }
    end = clock();
    cout << "LoadDataToGPUtime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;

    //SM��������10
    //ÿ���߳̿�Ĺ����ڴ��С��48 KB
    //ÿ���߳̿������߳�����1024
    //ÿ��SM������߳�����2048
    //ÿ��SM������߳�������64
    //����kernel��ִ������
    int blockx, blocky, gridx, gridy;
    blocky = 32;//�ߣ�����ΪQnum��������� 
    blockx = 32;//��blockx*blocky��32�ı�������С��64��������1024       
    gridy = (height + blocky - 1) / blocky;//�ߣ��̶�Ϊgridy*blocky=Qnum�ĳ��ȣ�gridy<2^31-1  
    gridx = (width + blockx - 1) / blockx;//��gridx*gridy���Ϊsm�ı���2�����ϣ���Ϊ32�ı�����ʵ�����ݲ�������32�ı���
    dim3 blockSize(blockx, blocky);//[32,32]
    dim3 gridSize(gridx, gridy);//[28, Qnums/32]

    //ִ��kernel//���Դ�����ʾ
    ComputeKernel << < gridSize, blockSize >> > (R, Q, C, dim, width);
    //ComputeKernel2 << < gridSize, blockSize >> > (R, Q, C, dim);
    // ͬ��device ��֤�������ȷ����
    cudaDeviceSynchronize();
    
    //for (size_t i = 0; i < C->width; i++)
    //{
    //    //C(row, col) = *(C.elements + row * C.width + col)
    //    if (C->elements[q0id*C->width + i]!=NULL)
    //    {
    //        std::cout << "ID��" << i << ";" << C->elements[q0id * C->width + i] << std::endl;
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
        if (vars.SearchF == 0)
            vars.VerifyAllCandidates((int)i,queries[i]);
        if (vars.SearchF == 1)
            vars.topkSearch((int)i,queries[i]);
        if (vars.SearchF == 2)
            vars.rangeSearch((int)i,queries[i]);
    }
 
    //    ID��0; 3.13513
    //    ID��1; 0.123433
    //    ID��2; 0.64123
    //    ID��3; 0.0110187
    //    ID��4; 0.0161557
    //    ID��5; 8.66706e-05
    //    ID��6; 15461.4
    //    ID��16; 0.000491897
    //    ID��17; 0.000741085
    //    ID��24; 0.565358
    //    ID��26; 4.15953
    //    ID��27; 0.00755924
    // 
    //    ID��2401; 0.065052
    //    ID��2402; -1
    //    ID��2403; 0.116964
    //    ID��2404; 0.579599
    //    ID��2405; 0.00066764
    //    ID��2406; 0.0495014
    //    ID��2407; -1
    //    ID��2408; 0.0199137
    //    ID��2409; 6.76879
    //    ID��2410; -1
    //    ID��2411; -1
    //    ID��2412; 0.279201
    //    ID��2413; -1
    //    ID��2414; 0.00328183
    //    ID��2415; 0.00267098
    return;
}


void CUDA_main()
{	
    int width = 1024;
    int height = 1024;
    Matrix* A, * B, * C;
    // �����й��ڴ�
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void**)&A->elements, nBytes);
    cudaMallocManaged((void**)&B->elements, nBytes);
    cudaMallocManaged((void**)&C->elements, nBytes);

    // ��ʼ������
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

    // ����kernel��ִ������
    dim3 blockSize(32, 32);//32*32
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);//(1024+31)/32=32.97
    // ִ��kernel//���Դ�����ʾ
    matMulKernel << < gridSize, blockSize >> > (A, B, C);


    // ͬ��device ��֤�������ȷ����
    cudaDeviceSynchronize();
    // ���ִ�н��
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    std::cout << "������: " << maxError << std::endl;
    return;
}

