#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

extern "C" void CUDACompute(char* argv[]);
extern "C" void CUDA_main();
extern "C" void CUDASingleQuery(char* argv[]);
extern "C" void getCudaInformaton();
#endif