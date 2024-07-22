#include "cpu_Header.h"

using namespace std;

bool Vars::ReadPara(char* argv[])
{   
    filePathEmb = argv[1];
    filePathTxt = argv[2];
    QNums = atoi(argv[3]);
    lambdaK=lambdaRange= atoi(argv[4]);
    SimF = atoi(argv[5]);
    if (SimF == 0 || SimF == 1 || SimF == 2 || SimF == 3)
        SimF = SimF;
    else
        return false;
    SearchF = atoi(argv[6]);

    if (SearchF == 0)
        k = atof(argv[7]);
    else if (SearchF == 1)
        range = (float)atof(argv[7]);
    else
        return false;
    return true;
}

void Vars::ShowPara()
{
    std::cout << "Record Nums: " << RNums << std::endl;
    std::cout << "Embedding Dimention: " << Dim << std::endl;
    std::cout << "Query Nums: " << QNums << std::endl;
    std::cout << "lambda: " << ((lambdaK > 0) ? lambdaK : lambdaRange) << std::endl;
    std::cout << "SimilarityFunc: " << (SimilarityF)SimF << std::endl;
    
    const char* SearchFstr = (SearchF == 0) ? "TopKSearch" : "RangeSearch";
    std::cout << "Search Type: " << SearchFstr << std::endl;
    const char* SearchKRstr = (SearchF == 0) ? "K: " : "Range: ";
    std::cout << SearchKRstr << ((SearchF == 0) ? k : range) << std::endl;
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
    step = (int)(Dim / 2);
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
            writeStr += to_string(j) + '\t' + to_string(candidateSets[i].cans[j].rID) + '\t' + to_string(candidateSets[i].cans[j].sim)+"\n";
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
        similarity = (float)overlap/max(q.size(), r.size());
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


void ComBoxIntersection(vector<float> q, vector<float> emb, int j, vector<candData>& cans, float qExpValue)
{
    int step =(int) emb.size() / 2;
    float boxInersection = 1.0;
    for (size_t i = 0; i < step; i++)
    {
        float tmp = min(q[i + step], emb[i + step]) - max(q[i], emb[i]);
        if (tmp <= 0)
        {
            boxInersection = -1.0;
            return;
        }
        boxInersection += (float)log(tmp);
    }
    candData cd;
    cd.rID = j;
    cd.sim = exp(boxInersection); //boxInersection regularization归一化到[0,1]区间?
    cans.push_back(cd);    
}

void ComBoxSim(vector<float> q, vector<float> emb, int j, vector<candData>& cans, SimilarityF f)
{
    int step = (int)emb.size() / 2;
    float sim = 0.0;
    float box_inersection = 0.0;
    float box_union = 0.0;
    float box_r = 0.0;
    float box_q = 0.0;
    for (size_t i = 0; i < step; i++)
    {
        float tmp_q = q[i + step] - q[i];
        float tmp_i = min(q[i + step], emb[i + step]) - max(q[i], emb[i]);
        float tmp_u = max(q[i + step], emb[i + step]) - min(q[i], emb[i]);
        float tmp_r = emb[i + step] - emb[i];
        if (tmp_i <= 0 || tmp_r<=0 || tmp_u<= 0|| tmp_q<=0)
        {
            sim = -1.0;
            return;
        }
        box_inersection += (float)log(tmp_i);
        box_union += (float)log(tmp_u);
        box_r += (float)log(tmp_r);
        box_q += (float)log(tmp_q);
    }

    if (f == Overlap)
        sim = exp(box_inersection - max(box_r, box_q));
    if (f == Jaccard)
        sim = exp(box_inersection - box_union);
    if (f == Cosine)
        sim = exp(box_inersection - (box_r+box_q)/2);
    if (f == Dice)
        sim = 2*exp(box_inersection)/ (exp(box_q) + exp(box_r) + 1e-10);

    candData cd;
    cd.rID = j;
    cd.sim = sim;
    cans.push_back(cd);
}

bool canLesser(candData& a, candData& b)
{
    return a.sim < b.sim;
}

bool canLarger(candData& a, candData& b)
{
    return a.sim > b.sim;
}

bool resLesser(resultData& a, resultData& b)
{
    return a.similarity < b.similarity;
}

bool resLarger(resultData& a, resultData& b)
{
    return a.similarity > b.similarity;
}

void Vars::SearchCandidates(int qID, SimilarityF f)
{
    candDatas candidateSet;
    candidateSet.qID = qID;

    //Compute qbox log value
    //float box_q = 1.0f;
    //for (size_t i = 0; i < embeddings[qID].size(); i++)
    //{
    //    float tmp = embeddings[qID][i + step] - embeddings[qID][i];
    //    if (tmp <= 0)
    //    {
    //        cout << "qi = " << qID  << "has a negatinve box" << endl;
    //        candidateSets.push_back(candidateSet);
    //        return;
    //    }
    //    else        
    //        box_q += log(tmp);        
    //    
    //}

    //Search Candidates for qi
    for (size_t j = 0; j < embeddings.size(); j++)
    {
        ComBoxSim(embeddings[qID], embeddings[j], (int)j, candidateSet.cans, f);
    }

    //Save qi's candidates
    candidateSets.push_back(candidateSet);
}

//Verify All Candidates for each qi
void Vars::VerifyAllCandidates(int id,int qID)
{
    resultDatas resultSet;
    resultSet.qID = qID;
        
    for (size_t i = 0; i < candidateSets[id].cans.size(); i++)
    {
        resultData result;
        int rID = result.rID = candidateSets[id].cans[i].rID;
        result.similarity = Similarity(SimF, orgText[qID].record, orgText[rID].record);
        resultSet.res.push_back(result);
    }    
    //Save qi's results
    resultSets3.push_back(resultSet);
}

//Verify All Candidates for each qi of topk search
void Vars::VerifyAllCandidatesForTopK(int id, int qID)
{
    resultDatas resultSet;
    resultSet.qID = qID;

    for (size_t i = 0; i < candidateSets[id].cans.size(); i++)
    {
        resultData result;
        int rID = result.rID = candidateSets[id].cans[i].rID;
        result.similarity = Similarity(SimF, orgText[qID].record, orgText[rID].record);
        
        if (resultSet.res.size() < k)
            resultSet.res.push_back(result);
        else
        {
            sort(resultSet.res.begin(), resultSet.res.end(), resLarger);
            if (result.similarity > resultSet.res[k - 1].similarity)
            {
                resultSet.res[k - 1].rID = result.rID;
                resultSet.res[k - 1].similarity = result.similarity;
            }
            else
                continue;
        }
    }
    //Save qi's results
    resultSets3.push_back(resultSet);
}

//Verify All Candidates for each qi of range search
void Vars::VerifyAllCandidatesForRange(int id, int qID)
{
    resultDatas resultSet;
    resultSet.qID = qID;

    for (size_t i = 0; i < candidateSets[id].cans.size(); i++)
    {
        resultData result;
        int rID = result.rID = candidateSets[id].cans[i].rID;
        result.similarity = Similarity(SimF, orgText[qID].record, orgText[rID].record);
        if (result.similarity >= range)
            resultSet.res.push_back(result); 
    }
    //Save qi's results
    resultSets3.push_back(resultSet);
}

void Vars::topkSearch(int id, int qID)
{    
    resultDatas resultSet;
    resultSet.qID = qID;

    //若embedding的similarity值可靠，candidate的顺序就是相似度的顺序
    //但embedding一定有误差，加入一个估计系数，计算前lambda*k个cans
    size_t sizeC = (size_t)candidateSets[id].cans.size();
    size_t sizeK = (size_t)ceil(k * lambdaK);
    if (sizeC > sizeK)
        sizeC = sizeK;
    for (size_t i = 0; i < sizeC; i++)
    {
        resultData result;
        int rID = result.rID = candidateSets[id].cans[i].rID;
        result.similarity = Similarity(SimF, orgText[qID].record, orgText[rID].record);
        
        if(resultSet.res.size()<k)
            resultSet.res.push_back(result);
        else
        {
            sort(resultSet.res.begin(), resultSet.res.end(), resLarger);
            if (result.similarity > resultSet.res[k-1].similarity)
            {
                resultSet.res[k-1].rID = result.rID;
                resultSet.res[k-1].similarity = result.similarity;
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

    //若embedding相交的值可靠，candidate的顺序就是相似度的顺序
    //但embedding一定有误差，加入一个估计系数，找到小于range的参数后，再继续计算lambda*i个cans
    size_t sizeC = (int)candidateSets[id].cans.size();
     
    for (size_t i = 0; i < sizeC; i++)
    {
        resultData result;
        int rID = result.rID = candidateSets[id].cans[i].rID;
        result.similarity = Similarity(SimF, orgText[qID].record, orgText[rID].record);
               
        if (result.similarity >= range)
            resultSet.res.push_back(result);
        else//再继续计算lambda*i个cans
        {
            size_t sizeR = (size_t)ceil(lambdaRange * i);
            if (sizeC > sizeR)
                sizeC = sizeR;
        }

    }
    //Save qi's results
    resultSets.push_back(resultSet);
}

//Estimate Search
void Vars::EstimateSearch(int id, int qID)
{
    EstimateLen = (int)(EstimateRate * Dim);
    float estimateQBox = 1.0f;
    int step = (int)(Dim / 2);
    //Q的box值作为估计值的估计部分
    for (size_t i = embeddings[qID].size(); i > embeddings[qID].size()- EstimateLen; i--)
    {        
        float tmp = embeddings[qID][i]- embeddings[qID][i-step];
        estimateQBox *= (float)abs(log10(tmp));
    }



}


float Vars::ComputeAccuracy(resultDatas& r, resultDatas& r3)
{
    vector<int> r3IDs;
    vector<int> rIDs;
    for (size_t i = 0; i < r3.res.size(); i++)
    {
        r3IDs.push_back(r3.res[i].rID);
    }
    for (size_t i = 0; i < r.res.size(); i++)
    {
        rIDs.push_back(r.res[i].rID);
    }
    sort(r3IDs.begin(), r3IDs.end());
    sort(rIDs.begin(), rIDs.end());

    int i = 0, j = 0, ans = 0;
    while (i < r3IDs.size() && j < rIDs.size()) {
        if (r3IDs[i] == rIDs[j]) ++i, ++j, ++ans;
        else {
            if (r3IDs[i] < rIDs[j]) ++i;
            else ++j;
        }
    }
    return ans*1.0f/ r3.res.size();
}


void CPUmain(char* argv[])
{
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
    start0=start = clock();
    //取QNums个随机记录作为查询Q
    srand((unsigned)2024);
    vector<int> qIDs;
    for (size_t n = 0; n < vars.QNums; n++)
    {
        int qid = rand() % vars.EmbNums;
        qIDs.push_back(qid);
        //cout << "QID = " << qid << endl;
    }
    
    //Search Candidates and Sort
    for (size_t i = 0; i < qIDs.size(); i++)
    {
        vars.SearchCandidates(qIDs[i], (SimilarityF)vars.SimF);
        //候选按照embed相交的大小排序
        //从小到大排
        //sort(vars.candidateSets[i].cans.begin(), vars.candidateSets[i].cans.end(), canLesser);
        //从大到小排
        if(vars.candidateSets[i].cans.size()>0)
            sort(vars.candidateSets[i].cans.begin(), vars.candidateSets[i].cans.end(), canLarger);
    }
    end = clock();
    cout << "CPUSearchTime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;

    start = clock();
    //Verify Results
    for (size_t i = 0; i < qIDs.size(); i++)
    {        
        if (vars.SearchF == 0)
            vars.topkSearch((int)i,qIDs[i]);
        if (vars.SearchF == 1)
            vars.rangeSearch((int)i,qIDs[i]);
    }
    end = clock();
    cout << "CPUVerifyTime = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    cout << "CPUtotalTime = " << double(end - start0) / CLOCKS_PER_SEC << "s" << endl;
    
    //Evaluation
    float tmpacc = 0;
    for (size_t i = 0; i < qIDs.size(); i++)
    {        
        if (vars.SearchF == 0)
            vars.VerifyAllCandidatesForTopK((int)i, qIDs[i]);
        if (vars.SearchF == 1)
            vars.VerifyAllCandidatesForRange((int)i, qIDs[i]);
        
        //resultSet Comparison
        float acc = vars.ComputeAccuracy(vars.resultSets[i], vars.resultSets3[i]);
        cout << "q" << to_string(i) << " Accuracy:" << to_string(acc) <<endl;
        vars.acccuacy.push_back(acc);       
        tmpacc += acc;
    }
    vars.AverageAcy = tmpacc / vars.QNums;
    cout << "Average Accuracy:" << to_string(vars.AverageAcy) << endl;
    //vars.SaveResults();
    return;
}