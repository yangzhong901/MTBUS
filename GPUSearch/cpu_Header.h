#pragma once

#ifndef CPU_HEADER_H
#define CPU_HEADER_H
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/timeb.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;

struct orgData
{
	int rID = 0;
	int rlen = 0;
	vector<int> record;
};

struct candData
{
	int rID = 0;
	float boxIntersection = -1.0;
};

struct candDatas
{
	int qID = 0;
	vector<candData> cans;
};

struct resultData
{
	int rID = 0;
	//int simFun = 0;
	float similarity = 0;
};

struct resultDatas
{
	int qID = 0;
	vector<resultData> res;
};

enum SimilarityF
{
	Overlap,
	Jaccard,
	Cosine,
	Dice
};

class Vars
{
public:
		
	int SimF = 0;//0,Overlap; 1,Jaccard; 2,Cosine; 3,Dice;
	int SearchF = 0;//0,topkSearch; 1,rangeSearch; 3, evaluation
	int k = 10;
	float range = 0.9f;
	float lambdaK = 1.1f;
	float lambdaRange = 1.1f;

	int RNums = -1;
	int Dim = -1;
	int EmbNums = -1;
	int QNums = -1;
	int Batch = 256;
	int LenOfEstimate = 1;

	char* filePathTxt = (char*)malloc(32 * sizeof(char));
	char* filePathEmb = (char*)malloc(32 * sizeof(char));

	vector<vector<float>> embeddings;
	vector<orgData> orgText;
	//candidates of all qis;
	vector<candDatas> candidateSets;
	//results of all qis;
	vector<resultDatas> resultSets;

	//candidates for evaluation;
	vector<candDatas> candidateSets3;
	//results for evaluation
	vector<resultDatas> resultSets3;

public:void ReadPara(char* argv[]);
public:void ShowPara();
public:void ReadEmbData();
public:void ReadOrgTextData();
public:float Similarity(int simFun, vector<int> q, vector<int> r);
public:void SearchCandidates(int qID);
public:void VerifyAllCandidates(int id, int qID);
public:void VerifyAllCandidatesForTopK(int id, int qID);
public:void	VerifyAllCandidatesForRange(int id, int qID);
public:void topkSearch(int id, int qID);
public:void rangeSearch(int id, int qID);
public:void SaveResults();
public:float ComputeAccuracy(resultDatas& r, resultDatas& r3);
};

void MatMul();
void CPUmain(char* argv[]);
#endif