# -*- coding: utf-8 -*-
"""
chatpdf.py
"""
import argparse
import hashlib
import os
import re
from threading import Thread
from typing import Union, List

import jieba
import torch
from loguru import logger
from peft import PeftModel
from similarities import (
    EnsembleSimilarity,
    BertSimilarity,
    BM25Similarity,
)
from similarities.similarity import SimilarityABC
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
    AutoModelForSequenceClassification,
)

jieba.setLogLevel("ERROR")

# 支持大模型字典
MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}
# Prompt编写
PROMPT_TEMPLATE = """基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

已知内容:
{context_str}

问题:
{query_str}
"""

"""
将长文本切分成固定长度的chunk
"""
class SentenceSplitter:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size  # 每个文本块（chunk）的最大字符数
        self.chunk_overlap = chunk_overlap  # 相邻文本块之间的重叠字符数,减少因分割而导致的上下文丢失
        # 思考一下，其实上下文丢失有两种情况：
        # 1. 在切分达到chunk_size后所遗留的后一部分，我们要把遗留的这后一部分也加进来。
        # 2. chunk本身就存在上下文语义丢失的情况，因此同样保留一个chunk的前后上下文。

    # 根据文本语言(中文或英文)调用对应的切分函数
    def split_text(self, text: str) -> List[str]:
        if self._is_has_chinese(text):
            return self._split_chinese_text(text)
        else:
            return self._split_english_text(text)

    # 利用jieba分词和标点符号对中文文本进行切分，形成一系列更小的文本块（chunks），同时尽量保持句子的完整性
    def _split_chinese_text(self, text: str) -> List[str]:
        sentence_endings = {'\n', '。', '！', '？', '；', '…'}  # 中文句末标点符号，用于判断句子是否完整性
        chunks, current_chunk = [], ''  # chunks 用于存储分割后的文本块，current_chunk 用于累积当前正在处理的文本块
        # 【遍历分词】
        for word in jieba.cut(text): # jieba对text分词
            # 超过chunk size
            if len(current_chunk) + len(word) > self.chunk_size:
                chunks.append(current_chunk.strip())  # 将当前的 current_chunk 添加到 chunks 列表中
                current_chunk = word # 直接赋值，相当于一个只包含的word新的current_chunk
            # 不超过chunk size，则继续追加
            else:
                current_chunk += word
            # 处理句末，逻辑是current_chunk 长度接近 self.chunk_size 且最后一个词 word 包含句末标点
            # 我们希望在句子结束的地方分割（即使 current_chunk 的长度略微超过 self.chunk_size）
            if word[-1] in sentence_endings and len(current_chunk) > self.chunk_size - self.chunk_overlap:
                chunks.append(current_chunk.strip())
                current_chunk = ''
        # 处理剩余文本（也就是添加的word不足以超过chunk size）
        if current_chunk:
            chunks.append(current_chunk.strip())
        # 处理块重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)
        return chunks

    # 直接利用正则表达式对英文文本进行句子切分
    def _split_english_text(self, text: str) -> List[str]:
        # 在每个句子结束的标点（. 句号，! 感叹号，或 ? 问号）之后，寻找一个或多个空白字符（\s+）
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        chunks, current_chunk = [], ''
        # 【遍历句子】
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size or not current_chunk:
                # current_chunk为空（即，第一个句子），则将当前句子添加到 current_chunk
                # current_chunk 不为空，在添加前会在句子前添加一个空格
                current_chunk += (' ' if current_chunk else '') + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        # 处理剩余文本
        if current_chunk:
            chunks.append(current_chunk)
        # 对chunk进行overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)

        return chunks

    # 判断文本是否包含中文字符
    def _is_has_chinese(self, text: str) -> bool:
        # check if contains chinese characters
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):  # 中文字符的编码范围\u4e00 - \u9fff
            return True
        else:
            return False

    # 在已经生成的文本块（chunks）之间创建重叠，减少上下文丢失
    def _handle_overlap(self, chunks: List[str]) -> List[str]:
        overlapped_chunks = []
        for i in range(len(chunks) - 1): # 忽略最后一个chunk，因为最后一个块后面不会有重叠
            # 拼接当前块和下一个块的overlap部分
            chunk = chunks[i] + ' ' + chunks[i + 1][:self.chunk_overlap]
            overlapped_chunks.append(chunk.strip())
        overlapped_chunks.append(chunks[-1])# 因为之前忽略了最后一个chunk，记得加上
        return overlapped_chunks

"""
基于RAG的聊天机器人,支持从多种格式的文档中检索信息并生成回复
"""


class ChatPDF:
    # 初始化各种模型,包括相似度模型、生成模型、rerank模型等
    """
    Init RAG model.
    :param similarity_model: 相似度模型，默认为 None，如果设置了此参数，则会使用它代替 EnsembleSimilarity（集成相似度模型）
    :param generate_model_type: 生成模型类型
    :param generate_model_name_or_path: 生成模型的名称或路径
    :param lora_model_name_or_path: LoRA（低秩适配）模型的名称或路径
    :param corpus_files: 语料库文件
    :param save_corpus_emb_dir: 保存语料库嵌入（embeddings）的目录，默认为 ./corpus_embs/
    :param device: 设备，默认为 None，自动选择 GPU 或 CPU
    :param int8: 使用 int8 量化，默认为 False
    :param int4: 使用 int4 量化，默认为 False
    :param chunk_size: 文本块大小，默认为 250
    # 控制文本分割的逻辑，决定是否再向后overlap
    :param chunk_overlap: 文本块重叠，默认为 0，如果 num_expand_context_chunk 大于 0，则不能设置为大于 0 的值
    :param rerank_model_name_or_path: 重排（rerank）模型的名称或路径，默认为 'BAAI/bge-reranker-base'
    :param enable_history: 是否启用历史，默认为 False
    # 控制生成模型在生成回答时考虑的上下文chunk
    :param num_expand_context_chunk: 扩展上下文块的数量，默认为 2，如果设置为 0，则不会扩展上下文块
    :param similarity_top_k: similarity_top_k，即相似度模型搜索 k 个语料库块，默认为 5
    # 通过打分来进行重排
    :param rerank_top_k: rerank_top_k，即重排模型搜索 k 个语料库块，默认为 3
    """
    def __init__(
            self,
            similarity_model: SimilarityABC = None,
            generate_model_type: str = "auto",
            generate_model_name_or_path: str = "01-ai/Yi-6B-Chat",
            lora_model_name_or_path: str = None,
            corpus_files: Union[str, List[str]] = None, # 一个文件str或多个文件List(str)
            save_corpus_emb_dir: str = "./corpus_embs/",
            device: str = None,
            int8: bool = False,
            int4: bool = False,
            chunk_size: int = 250,
            chunk_overlap: int = 0,
            rerank_model_name_or_path: str = None,
            enable_history: bool = False,
            num_expand_context_chunk: int = 2,
            similarity_top_k: int = 10,
            rerank_top_k: int = 3,
    ):

        if torch.cuda.is_available():
            default_device = torch.device(0)
        elif torch.backends.mps.is_available():
            default_device = torch.device('cpu')
        else:
            default_device = torch.device('cpu')
        self.device = device or default_device
        if num_expand_context_chunk > 0 and chunk_overlap > 0:
            logger.warning(f" 'num_expand_context_chunk' and 'chunk_overlap' cannot both be greater than zero. "
                           f" 'chunk_overlap' has been set to zero by default.")
            chunk_overlap = 0
        self.text_splitter = SentenceSplitter(chunk_size, chunk_overlap)
        if similarity_model is not None:
            self.sim_model = similarity_model
        else:
            # BertSimilarity（使用 BERT 模型来计算文本的语义相似度） 和 BM25Similarity（基于统计的相似度算法）
            # 词频（TF）：词在文档中出现的频率，表示重要性
            # 逆文档频率（IDF）：词在所有文档中的分布情况，表示罕见性
            # BM25 相对于 TF-IDF 的改进
            # 针对IDF
            # 长度归一化：BM25 通过参数 b 和 avgdl(D)（文档集合中文档的平均长度）考虑了文档长度的影响。
            # 较长的文档倾向于有更多的词出现，因此 BM25 通过长度归一化减少了这种偏差。
            # 针对TF
            # 控制词频的影响：BM25 通过参数 k_1 控制词频的影响，避免了词频过高导致的偏差。
            # 平滑处理：BM25 使用了一个平滑函数来处理词频，避免了文档中没有出现查询词时的除以零错误。
            m1 = BertSimilarity(model_name_or_path="shibing624/text2vec-base-multilingual", device=self.device)
            m2 = BM25Similarity()
            # c=2 是 BM25 算法中的一个参数，用于平衡词频和文档频率
            default_sim_model = EnsembleSimilarity(similarities=[m1, m2], weights=[0.5, 0.5], c=2)
            self.sim_model = default_sim_model
        self.gen_model, self.tokenizer = self._init_gen_model(
            generate_model_type,
            generate_model_name_or_path,
            peft_name=lora_model_name_or_path,
            int8=int8,
            int4=int4,
        )
        self.history = []
        self.corpus_files = corpus_files
        if corpus_files:
            self.add_corpus(corpus_files)
        self.save_corpus_emb_dir = save_corpus_emb_dir
        if rerank_model_name_or_path is None:
            rerank_model_name_or_path = "BAAI/bge-reranker-base"
        if rerank_model_name_or_path:
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name_or_path)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name_or_path)
            self.rerank_model.to(self.device)
            self.rerank_model.eval()
        else:
            self.rerank_model = None
            self.rerank_tokenizer = None
        self.enable_history = enable_history
        self.similarity_top_k = similarity_top_k
        self.num_expand_context_chunk = num_expand_context_chunk
        self.rerank_top_k = rerank_top_k

    def __str__(self):
        return f"Similarity model: {self.sim_model}, Generate model: {self.gen_model}"

    # 根据参数初始化生成模型和tokenizer
    def _init_gen_model(
            self,
            gen_model_type: str,
            gen_model_name_or_path: str,
            peft_name: str = None,
            int8: bool = False,
            int4: bool = False,
    ):
        """Init generate model."""
        if int8 or int4:
            device_map = None
        else:
            device_map = "auto"
        # 加载特定预训练模型
        model_class, tokenizer_class = MODEL_CLASSES[gen_model_type]
        tokenizer = tokenizer_class.from_pretrained(gen_model_name_or_path, trust_remote_code=True)
        model = model_class.from_pretrained(
            gen_model_name_or_path,
            load_in_8bit=int8 if gen_model_type not in ['baichuan', 'chatglm'] else False,
            load_in_4bit=int4 if gen_model_type not in ['baichuan', 'chatglm'] else False,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
        )
        # **回头看一下
        if self.device == torch.device('cpu'):
            model.float()
        if gen_model_type in ['baichuan', 'chatglm']:
            if int4:
                model = model.quantize(4).cuda()
            elif int8:
                model = model.quantize(8).cuda()
        try:
            model.generation_config = GenerationConfig.from_pretrained(gen_model_name_or_path, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load generation config from {gen_model_name_or_path}, {e}")
        # 加载lora
        if peft_name:
            model = PeftModel.from_pretrained(
                model,
                peft_name,
                torch_dtype="auto",
            )
            logger.info(f"Loaded peft model from {peft_name}")
        model.eval()
        return model, tokenizer

    # 将聊天历史转化为生成模型的输入格式
    def _get_chat_input(self):
        messages = []
        # 遍历对话历史
        for conv in self.history:
            if conv and len(conv) > 0 and conv[0]:
                messages.append({'role': 'user', 'content': conv[0]})
            if conv and len(conv) > 1 and conv[1]:
                messages.append({'role': 'assistant', 'content': conv[1]})
        # 应用对话模版
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,  # 对消息进行分词处理
            add_generation_prompt=True, # 添加生成提示
            return_tensors='pt' # 返回pytorch张量
        )
        # 将得到的输入张量 input_ids 移动到与生成模型 self.gen_model 相同的设备上
        return input_ids.to(self.gen_model.device)

    #流式生成回答
    @torch.inference_mode()
    # 不会计算梯度
    # batchnorm/dropout不再适应
    def stream_generate_answer(
            self,
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.0,
            context_len=2048
    ):
        # TextIteratorStreamer将生成的文本以迭代器的形式逐块提供
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = self._get_chat_input()
        # 在给定 上下文长度限制和最大新生成的token数的情况下，可以保留的输入序列的最大长度
        max_src_len = context_len - max_new_tokens - 8
        # 截取输入序列
        input_ids = input_ids[-max_src_len:]
        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
        # 启动多线程生成
        thread = Thread(target=self.gen_model.generate, kwargs=generation_kwargs)
        thread.start()
        # 将 streamer 产生的文本块逐一yield出来
        # yield 允许你惰性地生成值，这意味着只有在请求时才计算下一个值
        # yield from 在一个生成器函数中委托给另一个生成器函数
        yield from streamer

    # 添加语料库,支持多种文件格式,调用不同的extract_text函数提取文本并切分为chunk
    def add_corpus(self, files: Union[str, List[str]]):
        """Load document files."""
        if isinstance(files, str):
            files = [files]
        for doc_file in files:
            if doc_file.endswith('.pdf'):
                corpus = self.extract_text_from_pdf(doc_file)
            elif doc_file.endswith('.docx'):
                corpus = self.extract_text_from_docx(doc_file)
            elif doc_file.endswith('.md'):
                corpus = self.extract_text_from_markdown(doc_file)
            else:
                corpus = self.extract_text_from_txt(doc_file)  # 如果都是，则按照doc的方式执行
            # 使用换行符 \n 连接成一个长字符串 full_text
            full_text = '\n'.join(corpus)
            # chunk
            chunks = self.text_splitter.split_text(full_text)
            # 相似度模型会使用这些文本块来进行文本相似度计算
            self.sim_model.add_corpus(chunks)
        self.corpus_files = files
        logger.debug(f"files: {files}, corpus size: {len(self.sim_model.corpus)}, top3: "
                     f"{list(self.sim_model.corpus.values())[:3]}")

    # 计算一个或多个文件的 MD5 哈希值，可以产生一个 128 位（16 字节）的哈希值
    @staticmethod
    def get_file_hash(fpaths):
        hasher = hashlib.md5() # 用于计算 MD5 哈希值
        target_file_data = bytes()
        if isinstance(fpaths, str):
            fpaths = [fpaths]
        for fpath in fpaths:
            with open(fpath, 'rb') as file:
                chunk = file.read(1024 * 1024)  # read only first 1MB
                hasher.update(chunk)# 将chunk输入到haser中计算哈希值
                target_file_data += chunk

        hash_name = hasher.hexdigest()[:32] # 获取哈希对象的十六进制字符串表示，并取其前32位字符，作为文件的哈希值
        return hash_name

    # 从不同格式(pdf,txt,docx,md)的文件中提取文本，staticmethod可以独立于类的实例被调用
    @staticmethod
    def extract_text_from_pdf(file_path: str):
        """Extract text content from a PDF file."""
        import PyPDF2
        contents = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                ## 去空白
                page_text = page.extract_text().strip()
                # splitlines划分成多个文本行（根据\n划分），分别去除两端空白，并且只处理非空字符串
                raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
                # raw_text 中的元素仅仅是按行分割的文本片段，并不代表完整的句子或段落
                ## 从 raw_text 中重建出完整的句子
                new_text = ''
                for text in raw_text:
                    new_text += text
                    if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
                                    '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                        contents.append(new_text)
                        new_text = ''
                # 添加剩余文本
                if new_text:
                    contents.append(new_text)
        return contents

    @staticmethod
    def extract_text_from_txt(file_path: str):
        """Extract text content from a TXT file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            # 遍历每一行文本，去除两端空白字符
            contents = [text.strip() for text in f.readlines() if text.strip()]
        return contents

    @staticmethod
    def extract_text_from_docx(file_path: str):
        """Extract text content from a DOCX file."""
        import docx
        document = docx.Document(file_path)
        contents = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return contents

    @staticmethod
    def extract_text_from_markdown(file_path: str):
        """Extract text content from a Markdown file."""
        # 将markdown渲染成Beautiful Soup对象来解析 HTML 内容
        import markdown
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        html = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html, 'html.parser')
        contents = [text.strip() for text in soup.get_text().splitlines() if text.strip()]
        return contents

    # 为检索到的句子添加编号
    @staticmethod
    def _add_source_numbers(lst):
        """Add source numbers to a list of strings."""
        return [f'[{idx + 1}]\t "{item}"' for idx, item in enumerate(lst)]

    # 利用rerank模型对候选句子进行打分
    def _get_reranker_score(self, query: str, reference_results: List[str]):
        """Get reranker score."""
        pairs = []
        for reference in reference_results:
            pairs.append([query, reference])
        with torch.no_grad():
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            # 确保所有的输入数据都在正确的设备上
            inputs_on_device = {k: v.to(self.rerank_model.device) for k, v in inputs.items()}
            # 解包 inputs_on_device 字典，将其内容作为关键字参数传递给模型
            # 重塑为一维向量，-1 表示自动计算该维度的大小，以便保持元素总数不变
            # 将 logits 张量转换为浮点数格式
            scores = self.rerank_model(**inputs_on_device, return_dict=True).logits.view(-1, ).float()

        return scores

    # 检索与query相关的句子,并进行rerank和扩展上下文
    def get_reference_results(self, query: str):
        """
        Get reference results.
            1. Similarity model get similar chunks
            2. Rerank similar chunks
            3. Expand reference context chunk
        :param query:
        :return:
        """
        reference_results = []## 用于存储与查询 query 最相似的文本块，也用于后续reranker
        sim_contents = self.sim_model.most_similar(query, topn=self.similarity_top_k)
        # 返回的例子{
        #     'query1': {
        #         'corpus_id1': 0.85,
        #         'corpus_id2': 0.75,
        #         'corpus_id3': 0.65
        #     },
        #     'query2': {
        #         'corpus_id2': 0.80,
        #         'corpus_id4': 0.90
        #     }
        # }
        # Get reference results from corpus
        hit_chunk_dict = dict()## 用于存储文本块的索引（由 corpus_id 表示）和对应的文本块内容（hit_chunk），便于后续添加上下文索引
        # 得到similar模型计算的与query相似的chunk
        for query_id, id_score_dict in sim_contents.items():
            for corpus_id, s in id_score_dict.items():
                hit_chunk = self.sim_model.corpus[corpus_id]
                reference_results.append(hit_chunk)
                hit_chunk_dict[corpus_id] = hit_chunk

        if reference_results:
            # 使用reranker
            if self.rerank_model is not None:
                # Rerank reference results
                rerank_scores = self._get_reranker_score(query, reference_results)
                logger.debug(f"rerank_scores: {rerank_scores}")
                # Get rerank top k chunks
                # 将 reference_results 列表和 rerank_scores 列表zip成一个元组列表
                # 指定排序的依据是元组中的第二个元素（即重排分数）降序
                # 通过切片取前K个元素
                reference_results = [reference for reference, score in sorted(
                    zip(reference_results, rerank_scores), key=lambda x: x[1], reverse=True)][:self.rerank_top_k]
                # 检查当前的 hit_chunk 是否存在于重排后的 reference_results 列表中
                # 如果 hit_chunk 存在于 reference_results 中，更新 hit_chunk_dict 字典，只保留那些被选为参考结果的文本块
                hit_chunk_dict = {corpus_id: hit_chunk for corpus_id, hit_chunk in hit_chunk_dict.items() if
                                  hit_chunk in reference_results}
            # Expand reference context chunk
            if self.num_expand_context_chunk > 0:
                new_reference_results = []
                # 遍历命中块字典
                for corpus_id, hit_chunk in hit_chunk_dict.items():
                    # 获取前一个文本块,get corpus_id - 1，不存在则为空字符串
                    expanded_reference = self.sim_model.corpus.get(corpus_id - 1, '') + hit_chunk
                    # 获得后n个文本块
                    for i in range(self.num_expand_context_chunk):
                        expanded_reference += self.sim_model.corpus.get(corpus_id + i + 1, '')
                    new_reference_results.append(expanded_reference)
                # 扩展上下文后的参考文本块
                reference_results = new_reference_results
        return reference_results

    # 流式预测接口：使用 yield 来逐步产生文本
    # TODO:存储历史对话
    def predict_stream(
            self,
            query: str,
            max_length: int = 512,
            context_len: int = 2048,
            temperature: float = 0.7,
    ):
        """Generate predictions stream."""
        # 用于识别生成文本的结束
        stop_str = self.tokenizer.eos_token if self.tokenizer.eos_token else "</s>"
        # 不使用历史记录
        if not self.enable_history:
            self.history = []
        # RAG
        if self.sim_model.corpus:
            # 获取最相关的块
            reference_results = self.get_reference_results(query)
            if not reference_results:
                yield '没有提供足够的相关信息', reference_results
            # 有参考结果，添加源编号，构造提示文本 prompt 并记录到日志
            reference_results = self._add_source_numbers(reference_results)
            context_str = '\n'.join(reference_results)[:(context_len - len(PROMPT_TEMPLATE))]
            prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=query)
            logger.debug(f"prompt: {prompt}")
        # 非RAG
        else:
            prompt = query
            logger.debug(prompt)
        self.history.append([prompt, ''])
        # 使用 stream_generate_answer 方法开始生成回复
        response = ""
        for new_text in self.stream_generate_answer(
                max_new_tokens=max_length,
                temperature=temperature,
                context_len=context_len,
        ):
            # 如果生成的文本不是结束字符串，则将其追加到 response 并 yield 出去
            if new_text != stop_str:
                response += new_text
                # self.history[-1][1] = response # 是否可行？
                yield response

    # 非流式预测接口：累积整个文本后再返回
    def predict(
            self,
            query: str,
            max_length: int = 512,
            context_len: int = 2048,
            temperature: float = 0.7,
    ):
        """Query from corpus."""
        reference_results = []
        if not self.enable_history:
            self.history = []
        if self.sim_model.corpus:
            reference_results = self.get_reference_results(query)

            if not reference_results:
                return '没有提供足够的相关信息', reference_results
            reference_results = self._add_source_numbers(reference_results)
            context_str = '\n'.join(reference_results)[:(context_len - len(PROMPT_TEMPLATE))]
            prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=query)
            logger.debug(f"prompt: {prompt}")
        else:
            prompt = query
        self.history.append([prompt, ''])
        response = ""
        # 非流式用for循环
        for new_text in self.stream_generate_answer(
                max_new_tokens=max_length,
                temperature=temperature,
                context_len=context_len,
        ):
            response += new_text
        response = response.strip()
        self.history[-1][1] = response  # 存到最近一次交互的第二个位置上（第二个位置是因为：第一个元素是用户的话语，第二个元素是助手的回应）
        return response, reference_results

    # 保存语料库的embedding
    def save_corpus_emb(self):
        dir_name = self.get_file_hash(self.corpus_files)
        save_dir = os.path.join(self.save_corpus_emb_dir, dir_name)
        if hasattr(self.sim_model, 'save_corpus_embeddings'):
            self.sim_model.save_corpus_embeddings(save_dir)
            logger.debug(f"Saving corpus embeddings to {save_dir}")
        return save_dir

    # 加载预先计算好的语料库embedding
    def load_corpus_emb(self, emb_dir: str):
        # hasattr 函数用于检查一个对象是否拥有特定的属性
        if hasattr(self.sim_model, 'load_corpus_embeddings'):
            logger.debug(f"Loading corpus embeddings from {emb_dir}")
            self.sim_model.load_corpus_embeddings(emb_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_model_name", type=str, default="shibing624/text2vec-base-multilingual")
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model_name", type=str, default="01-ai/Yi-6B-Chat")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--rerank_model_name", type=str, default="")
    parser.add_argument("--corpus_files", type=str, default="sample.pdf")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--int4", action='store_true', help="use int4 quantization")
    parser.add_argument("--int8", action='store_true', help="use int8 quantization")
    parser.add_argument("--chunk_size", type=int, default=220)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--num_expand_context_chunk", type=int, default=1)
    args = parser.parse_args()
    print(args)
    sim_model = BertSimilarity(model_name_or_path=args.sim_model_name, device=args.device)
    m = ChatPDF(
        similarity_model=sim_model,
        generate_model_type=args.gen_model_type,
        generate_model_name_or_path=args.gen_model_name,
        lora_model_name_or_path=args.lora_model,
        device=args.device,
        int4=args.int4,
        int8=args.int8,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        corpus_files=args.corpus_files.split(','),
        num_expand_context_chunk=args.num_expand_context_chunk,
        rerank_model_name_or_path=args.rerank_model_name,
    )
    r, refs = m.predict('自然语言中的非平行迁移是指什么？')
    print(r, refs)
