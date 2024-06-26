- 本项目支持多种开源LLM模型，包括ChatGLM3-6b、Chinese-LLaMA-Alpaca-2、Baichuan、YI等
- 本项目支持多种文件格式，包括PDF、docx、markdown、txt等
- 本项目优化了RAG准确率
  - Chinese chunk切分优化，适配中英文混合文档
  - embedding优化，使用text2vec的sentence embedding，支持sentence embedding/字面相似度匹配算法
  - 检索匹配优化，引入jieba分词的rank_BM25，提升对query关键词的字面匹配，使用字面相似度+sentence embedding向量相似度加权获取corpus候选集
  - 新增reranker模块，对字面+语义检索的候选集进行rerank排序，减少候选集，并提升候选命中准确率，用`rerank_model_name_or_path`参数设置rerank模型
  - 新增候选chunk扩展上下文功能，用`num_expand_context_chunk`参数设置命中的候选chunk扩展上下文窗口大小
  - RAG底模优化，可以使用200k的基于RAG微调的LLM模型，支持自定义RAG模型，用`generate_model_name_or_path`参数设置底模
- 本项目基于gradio开发了RAG对话页面，支持流式对话
## 数据来源
1. medical_corpus.jsonl：https://github.com/shibing624/MedicalGPT/blob/main/data/rag/medical_corpus.txt
2. sample.pdf：https://arxiv.org/abs/1705.09655
## 使用说明

#### 安装依赖

```shell
pip install -r requirements.txt
```

#### 本地调用

```shell
CUDA_VISIBLE_DEVICES=0 python chatpdf.py --gen_model_type auto --gen_model_name 01-ai/Yi-6B-Chat --corpus_files sample.pdf
```

#### 启动Web服务

```shell
CUDA_VISIBLE_DEVICES=0 python webui.py --gen_model_type auto --gen_model_name 01-ai/Yi-6B-Chat --corpus_files sample.pdf --share
```

浏览器地址栏中 http://localhost:7860 

