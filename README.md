# DocSeeker: 基于多模态检索工具增强的文档理解智能体
## 原始实验结果公开
|实验名称|实验结果路径|
|-------|-----------|
|DocSeeker框架|`./src/Mybenchmark/Qwen3vl8b`|
|对比实验|`./src/compare_experiment`|
|消融实验|`./src/ablation_text_only`,`./src/ablation_page_only`|
## 环境配置
由于Qwen3VL的VLLM/Transformer库推理与Colpali所需的环境**有着不可解决的冲突**，因此需要各自配置隔离的环境。
```
conda create -n qwen python=3.10   # 用于Qwen3vl推理
conda create -n colpali python=3.10 # 用于colpali
conda activate qwen
pip install -r requirements1.txt
conda activate colpali
pip install -r requirements2.txt
```

## 模型下载
上huggingface或modelscope下载至少以下模型：
- Qwen3-VL-8B-Instruct
- Qwen3-Embedding-0.6B
- colqwen2.5-v0.2
- colqwen2.5-base
## 数据集下载
`https://huggingface.co/datasets/VRRRRR/DocSeeker-Bench/tree/main`

## 单样本推理
首先启动colpali服务
```
cd src
conda activate colpali
python service/server.py \
    --model /HOME/sysu_gbli2/sysu_gbli2xy_1/HDD_POOL/zdw/Docproject/colqwen2.5-v0.2 \
    --device cuda:0 \
    --port 8788
```
在`config.py`中调整你的推理backbone，并修改地址

随后进行单样本推理
```
python pipeline.py \
    --pdf_path "/path/to/your/pdf" \
    --question "In Figure 3, comparing the performance of EvoComp (l = 0) and EvoComp (l = 0) Transferred on the GQA benchmark, which statement best describes their relative performance at a 70.0% compression rate?"
```

## 在DocSeeker-Bench上进行评测
- 在huggingface仓库`https://huggingface.co/datasets/VRRRRR/DocSeeker-Bench/tree/main`下载数据
- 将pdf文件放到`./src/Mybenchmark/data`文件夹
- 将`question.json`放到`./src/Mybenchmark/question.json`

首先启动colpali服务
```
cd src
conda activate colpali
python service/server.py \
    --model /HOME/sysu_gbli2/sysu_gbli2xy_1/HDD_POOL/zdw/Docproject/colqwen2.5-v0.2 \
    --device cuda:0 \
    --port 8788
```

评测
```
python evaluate.py \
    --output_dir /path/to/output_dir
```

## Benchmark数据生成
将每个pdf提取为image，放在`./gen_data/paper_img/{pdf_name}/{page_idx}.jpg`，`page_idx`从0开始

```
cd gen_data
python gen_benchmark.py
```

