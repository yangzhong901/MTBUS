# Soure Code for Universal Set Similarity Search via Multi-Task Representation Learning
## Settings
python 3.10
## Usage
### MTB Example
python main_set2box_mutitask.py --dataset gplus --gpu 0 --epochs 100 --dim 64 --learning_rate 1e-3
### Usearch Example
CUDATest.exe embedding_data_path original_token_path query_number lamda similarity_function search_type k/range CPU/GPU
similarity_function: 0-Overlap; 1-Jaccard; 2-Cosine; 3-Dice;
search_type:0-topkSearch; 1-rangeSearch;
1. Run a top-k query with k = 100, query_number = 30
CPU version:
..\data\gplusMT_embeds_16.txt ..\data\gplus\test.txt 30 50 1 0 100 1
GPU version:
..\data\gplusMT_embeds_16.txt ..\data\gplus\test.txt 30 50 1 0 100 0
