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
	float sim = -1.0;
};

//candidates for qi
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

struct candMultiData
{
	int rID = 0;
	float ov = -1.0;
	float ja = -1.0;
	float cs = -1.0;
	float di = -1.0;
};

struct candMultiDatas
{
	int qID = 0;
	vector<candMultiData> cans;
};
struct resultOverlap
{
	int rID = 0;	
	float value = 0;
};
struct resultJaccard
{
	int rID = 0;
	float value = 0;
};
struct resultCosine
{
	int rID = 0;
	float value = 0;
};
struct resultDice
{
	int rID = 0;
	float value = 0;
};
struct resultMultiDatas
{
	int qID = 0;
	vector<resultOverlap> resOv;
	vector<resultJaccard> resJa;
	vector<resultCosine> resCs;
	vector<resultDice> resDi;
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
	int k = 50;
	float range = 0.9f;
	float lambdaK = 1.0f;
	float lambdaRange = 1.0f;

	int RNums = -1;
	int Dim = -1;
	int step = 1;
	int EmbNums = -1;
	int QNums = -1;
	int Batch = 256;
	float EstimateRate = 0.25f;
	int EstimateLen = 1;

	char* filePathTxt = (char*)malloc(32 * sizeof(char));
	char* filePathEmb = (char*)malloc(32 * sizeof(char));

	vector<vector<float>> embeddings;
	vector<orgData> orgText;
	//candidates of all qis;
	vector<candDatas> candidateSets;
	vector<candMultiDatas> candiMultidateSets;
	//results of all qis;
	vector<resultDatas> resultSets;
	vector<resultOverlap> resultOverlapSets;
	vector<resultJaccard> resultJaccardSets;
	vector<resultCosine> resultCosineSets;
	vector<resultDice> resultDiceSets;
	//candidates for evaluation;
	vector<candDatas> candidateSets3;
	//results for evaluation
	vector<resultDatas> resultSets3;

	//accuracy 
	vector<float> acccuacy;
	//average accuracy
	float AverageAcy = 0.0f;;

public:bool ReadPara(char* argv[]);
public:void ShowPara();
public:void ReadEmbData();
public:void ReadOrgTextData();
public:float Similarity(int simFun, vector<int> q, vector<int> r);

public:void SearchCandidates(int qID, SimilarityF f);
public:void VerifyAllCandidates(int id, int qID);
public:void VerifyAllCandidatesForTopK(int id, int qID);
public:void	VerifyAllCandidatesForRange(int id, int qID);
public:void topkSearch(int id, int qID);
public:void rangeSearch(int id, int qID);
public:void SaveResults();
public:float ComputeAccuracy(resultDatas& r, resultDatas& r3);
public:void EstimateSearch(int id, int qID);

};

void MatMul();
void CPUmain(char* argv[]);
#endif