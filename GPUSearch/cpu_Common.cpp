#include "cpu_Header.h"

using namespace std;

void Vars::ReadPara(char* argv[])
{    
    filePathEmb = argv[1];
    filePathTxt = argv[2];
    QNums = atoi(argv[3]);
    Batch = atoi(argv[4]);
    SimF = atoi(argv[5]);
    SearchF = atoi(argv[6]);
    if (SearchF==0)
        k= atoi(argv[7]);
    else if (SearchF == 1)
        range = atof(argv[7]);
}

void Vars::ShowPara()
{
    std::cout << "Record Nums：" << RNums << std::endl;
    std::cout << "Embedding Dimention：" << Dim << std::endl;
    std::cout << "Query Nums：" << QNums << std::endl;
    std::cout << "Similarity：" << (SimilarityF)SimF << std::endl;
    std::cout << "Search Type：" << SearchF << std::endl;
}

void Vars::ReadEmbData()
{
    char* filename = filePathEmb;
    int lineNum = 0;     
    string line;
    embeddings.clear();
   
    ifstream infile(filename);
    assert(infile.is_open());

    if (infile.is_open())
    {
        while (getline(infile, line))
        {
            vector<float> lineEmbs;
            stringstream ss(line);
            string str_tmp;
            while (getline(ss, str_tmp, '\t'))
            {
                float val = stof(str_tmp);
                lineEmbs.push_back(val);
            }
            embeddings.push_back(lineEmbs);
            lineNum++;
            EmbNums = lineNum;
            line.clear();
        }        
    }
    Dim = (int)embeddings[0].size();
    cout<< "# Embeddings Records: " << lineNum << endl;
}

void Vars::ReadOrgTextData()
{
    char* filename = filePathTxt;
    int lineNum = 0;
    ifstream infile(filename);
    string line;
    
    int len = 0;
    int totalLen = 0;
    int minLen = 0xffff;
    int maxLen = 0;

    orgText.clear();
    if (infile.is_open())
    {
        getline(infile, line);
        infile.clear();//忽略第一行
        while (getline(infile, line))
        {            
            stringstream ss(line);
            string str_tmp;
            orgData od_tmp;
            while (getline(ss, str_tmp, '\t'))
            {
                int val = stoi(str_tmp);
                od_tmp.record.push_back(val);
            }

            len = (int)od_tmp.record.size();
            od_tmp.rlen = len;
            od_tmp.rID = lineNum;
            orgText.push_back(od_tmp);

            totalLen += len;
            if (len>maxLen)
            {
                maxLen = len;
            }
            if (len<minLen)
            {
                minLen = len;
            }
            
            lineNum++;
            RNums = lineNum;
            line.clear();
        }
    }
    cout << "# String Records: " << lineNum << endl;
    cout << "# Minmum Record Size : " << minLen << endl;
    cout << "# Maximum Record Size: " << maxLen << endl;
    cout << "# Average Record Size :" << ceil(totalLen/lineNum) << endl;
}


void Vars::SaveResults()
{    
    int qid = resultSets[0].qID;
    string searchFstr = "-K" + k;
    if (SearchF == 1)
        searchFstr = "-Range"+ to_string(range);
    //save candidates
    string canfilename = "./data/ml1m_results/Embed" + to_string(EmbNums) + "-Qnum" + to_string(QNums) + "-Sim"
        + to_string(SimF) + "-Type" + to_string(SearchF) + searchFstr + "Q" + to_string(qid) + "-Cans.txt";
    ofstream outcans;
    outcans.open(canfilename,ios::out|ios::trunc);
    if (!outcans.is_open())
    {
        cout<< "Openging Error" << endl;
        return;
    }
    for (size_t i = 0; i < candidateSets.size(); i++)
    {
        string writeStr = to_string(candidateSets[i].qID) + '\n';
        for (size_t j = 0; j < candidateSets[i].cans.size(); j++)
        {
            writeStr += to_string(j) + '\t' + to_string(candidateSets[i].cans[j].rID) + '\t' + to_string(candidateSets[i].cans[j].boxIntersection)+"\n";
        }
        
        const char* buffer = writeStr.data();
        int len = strlen(buffer);
        outcans.write(buffer, sizeof(char)*len);
        outcans.flush();
    }
    outcans.close();
    //save results
    string resfilename = "./data/ml1m_results/Embed" + to_string(EmbNums) + "-Qnum" + to_string(QNums) + "-Sim"
        + to_string(SimF) + "-Type" + to_string(SearchF) + searchFstr + "Q" + to_string(qid) + "-Res.txt";
    ofstream rescans;
    rescans.open(resfilename, ios::out | ios::trunc);
    for (size_t i = 0; i < resultSets.size(); i++)
    {
        string writeStr =  to_string(resultSets[i].qID) + '\n';
        for (size_t j = 0; j < resultSets[i].res.size(); j++)
        {
            writeStr += to_string(j) + '\t' + to_string(resultSets[i].res[j].rID) +'\t'+ to_string(resultSets[i].res[j].similarity) + "\n";
        }

        const char* buffer = writeStr.data();
        int len = strlen(buffer);
        rescans.write(buffer, sizeof(char) * len);
        rescans.flush();
    }
    rescans.close();
}

int CountOverlap(vector<int>& xArray, vector<int>& yArray, int xLen, int yLen, int xStart, int yStart, int requiredOverlap)
{
    int i = xStart, j = yStart, ans = 0;
    while (i < xLen && j < yLen) {
        if (xArray[i] == yArray[j]) ++i, ++j, ++ans;
        else {
            if (xArray[i] < yArray[j]) ++i;
            else ++j;
        }
    }
    if (ans < requiredOverlap) return INT_MIN;
    else return ans;
}

float Vars::Similarity(int simFun, vector<int> q, vector<int> r)
{
    sort(q.begin(), q.end());
    sort(r.begin(), r.end());

    float similarity = 0;
    int overlap = 0;
    int indexq = 0;
    int indexr = 0;
    //count overlap
    while (indexq < q.size() && indexr < r.size())
    {
        if (q[indexq] == r[indexr])
        {
            ++indexq, ++indexr, ++overlap;
        }
        else
        {
            if (q[indexq] < r[indexr])
                ++indexq;
            else
                ++indexr;
        }
    }
    if (simFun == Overlap)
        similarity = (float)overlap;
    else if (simFun == Jaccard)
        similarity = overlap * 1.0f / (q.size() + r.size() - overlap);
    else if (simFun == Cosine)
        similarity = overlap * 1.0f / sqrt(q.size() * r.size());
    else if (simFun == Dice)
        similarity = overlap * 2.0f / (q.size() + r.size());
    else
        similarity = -INFINITY;
    return similarity;
}


void MatMul()
{
    int height = 1024;
    int width = 1024;

    //初始化
    int** A = new int* [width];//A[a][b]
    int** B = new int* [width];//B[b][a]
    int** C = new int* [width];//C[m][n]=C[a][a]

    for (int i = 0; i < height; i++)
    {
        A[i] = new int[height];
        B[i] = new int[height];
        C[i] = new int[height];
    }

    //赋值
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            A[i][j] = 1;
            B[i][j] = 2;
        }
    }
    //矩阵相乘
    for (int m = 0; m < height; m++) {
        for (int n = 0; n < height; n++)
        {
            int cvaule = 0;
            for (int b = 0; b < width; b++)
            {
                cvaule += A[m][b] * B[b][n];
            }
            C[m][n] = cvaule;
        }
    }
}



void ComBoxIntersection(vector<float> q, vector<float> emb, int j, vector<candData>& cans)
{
    int step =(int) emb.size() / 2;
    float boxInersection = 1.0;
    for (size_t i = 0; i < step; i++)
    {
        double tmp = min(q[i+step],emb[i + step]) - max(q[i], emb[i]);
        if(tmp<=0)
        {
            boxInersection = -1.0;
            return;
        }
        //log regularization
        boxInersection *= (float)abs(log10(tmp));
    }
    candData cd;
    cd.rID = j;
    cd.boxIntersection = boxInersection;
    cans.push_back(cd);    
}

bool canLesser(candData& a, candData& b)
{
    return a.boxIntersection < b.boxIntersection;
}

bool canLarger(candData& a, candData& b)
{
    return a.boxIntersection > b.boxIntersection;
}

bool resLesser(resultData& a, resultData& b)
{
    return a.similarity < b.similarity;
}

bool resLarger(resultData& a, resultData& b)
{
    return a.similarity > b.similarity;
}

void Vars::SearchCandidates(int qID)
{
    candDatas candidateSet;
    candidateSet.qID = qID;
    //Search Candidates for qi
    for (size_t j = 0; j < embeddings.size(); j++)
    {
        //if(j==1816)
        ComBoxIntersection(embeddings[qID], embeddings[j], (int)j, candidateSet.cans);
    }
    //for (size_t c = 0; c < candidateSet.cans.size(); c++)
    //{
    //    std::cout << "ID：" << candidateSet.cans[c].rID << ";" << candidateSet.cans[c].boxIntersection << std::endl;
    //}
    //Save qi's candidates
    candidateSets.push_back(candidateSet);
}

//Search and Verify All Candidates for qi
void Vars::VerifyAllCandidates(int id,int qID)
{   
    resultDatas resultSet;
    resultSet.qID = qID;
    //候选按照embed相交的大小排序
    sort(candidateSets[id].cans.begin(), candidateSets[id].cans.end(), canLesser);
    for (size_t i = 0; i < candidateSets[id].cans.size(); i++)
    {
        resultData result;
        int rID = result.rID = candidateSets[id].cans[i].rID;
        result.similarity = Similarity(SimF, orgText[qID].record, orgText[rID].record);
        resultSet.res.push_back(result);
    }    
    //Save qi's results
    resultSets.push_back(resultSet);
}

void Vars::topkSearch(int id, int qID)
{    
    resultDatas resultSet;
    resultSet.qID = qID;
   
    //候选按照embed相交的大小排序
    //从小到大排,log正则化之后，值越小越相似
    sort(candidateSets[id].cans.begin(), candidateSets[id].cans.end(), canLesser);
    //从大到小排
    //sort(candidateSet.cans.begin(), candidateSet.cans.end(),greater<float>());

    //若embedding相交的值可靠，candidate的顺序就是相似度的顺序
    //但embedding一定有误差，加入一个估计系数，计算前lambda*k个cans
    for (size_t i = 0; i < ceil(lambdaK*k); i++)
    {
        resultData result;
        int rID = result.rID = candidateSets[id].cans[i].rID;
        result.similarity = Similarity(SimF, orgText[qID].record, orgText[rID].record);
        
        if(resultSet.res.size()<k)
            resultSet.res.push_back(result);
        else
        {
            sort(resultSet.res.begin(), resultSet.res.end(), resLarger);
            if (result.similarity > resultSet.res[k].similarity)
            {
                resultSet.res[k].rID = result.rID;
                resultSet.res[k].similarity = result.similarity;
            }
            else
                continue;
        }
    }
    sort(resultSet.res.begin(), resultSet.res.end(), resLarger);

    //Save qi's results
    resultSets.push_back(resultSet);
}

void Vars::rangeSearch(int id, int qID)
{   
    resultDatas resultSet;
    resultSet.qID = qID;
    //Search Candidates for qi
    
    //候选按照embed相交的大小排序
    //从小到大排,log正则化之后，值越小越相似
    sort(candidateSets[id].cans.begin(), candidateSets[id].cans.end(), canLesser);
    //从大到小排
    //sort(candidateSet.cans.begin(), candidateSet.cans.end(),greater<float>());

    //若embedding相交的值可靠，candidate的顺序就是相似度的顺序
    //但embedding一定有误差，加入一个估计系数，找到小于range的参数后，再继续计算lambda*i个cans
    int sizeR = (int)embeddings.size();
    for (size_t i = 0; i < sizeR; i++)
    {
        resultData result;
        int rID = result.rID = candidateSets[id].cans[i].rID;
        result.similarity = Similarity(SimF, orgText[qID].record, orgText[rID].record);
               
        if (result.similarity >= range)
            resultSet.res.push_back(result);
        else//再继续计算lambda*i个cans
            sizeR = (int)ceil(lambdaRange * i);

    }
    //Save qi's results
    resultSets.push_back(resultSet);
}

void CPUmain(char* argv[])
{
    Vars vars;
    vars.ReadPara(argv);
    vars.ReadOrgTextData();
    vars.ReadEmbData();
    vars.ShowPara();

    //取QNums个随机记录作为查询Q
    srand((unsigned)2024);
    vector<int> qIDs;
    for (size_t n = 0; n < vars.QNums; n++)
    {
        int qid = rand() % vars.EmbNums;
        qIDs.push_back(qid);
        //cout << "QID = " << qid << endl;
    }
    
    for (size_t i = 0; i < qIDs.size(); i++)
    {
        vars.SearchCandidates(qIDs[i]);
        if (vars.SearchF == 0)
            vars.VerifyAllCandidates((int)i,qIDs[i]);
        if (vars.SearchF == 1)
            vars.topkSearch((int)i,qIDs[i]);
        if (vars.SearchF == 2)
            vars.rangeSearch((int)i,qIDs[i]);
    }

    vars.SaveResults();
    return;
}