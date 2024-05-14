# -*- coding: utf-8 -*-
"""
batch_rag_demo.py
批量问答生成和评估的示例脚本
1.加载给定的语料库文件,构建 ChatPDF 模型。
2.读取测试查询文件或使用默认查询。
3.对每个查询,使用 ChatPDF 模型生成回答,并将生成的回答与真实答案进行比较。
4.将生成的回答、参考结果和真实答案保存到输出的 JSONL 文件中。
5.评估生成速度和效率。
"""
import argparse
import json
import os
import time

from similarities import BM25Similarity
from tqdm import tqdm

from chatpdf import ChatPDF

pwd_path = os.path.abspath(os.path.dirname(__file__))


# 从JSONL文件中解析出问答对,构建ground truth字典
def get_truth_dict(jsonl_file_path):
    truth_dict = dict()
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        # 从中解析出问题和答案,构建一个问题到答案的字典映射
        for line in file:
            entry = json.loads(line)
            input_text = entry.get("question", "")
            output_text = entry.get("answer", "")
            if input_text and output_text:
                truth_dict[input_text] = output_text

    return truth_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model_name", type=str, default="01-ai/Yi-6B-Chat")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--rerank_model_name", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--corpus_files", type=str, default="medical_corpus.jsonl")
    parser.add_argument('--query_file', default="medical_query.txt", type=str, help="query file, one query per line")
    parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument("--int4", action='store_true', help="use int4 quantization")
    parser.add_argument("--int8", action='store_true', help="use int8 quantization")
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--num_expand_context_chunk", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--test_size", type=int, default=-1)
    args = parser.parse_args()
    print(args)
    sim_model = BM25Similarity()
    model = ChatPDF(
        similarity_model=sim_model,
        generate_model_type=args.gen_model_type,
        generate_model_name_or_path=args.gen_model_name,
        lora_model_name_or_path=args.lora_model,
        corpus_files=args.corpus_files.split(','),
        device=args.device,
        int4=args.int4,
        int8=args.int8,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rerank_model_name_or_path=args.rerank_model_name,
        num_expand_context_chunk=args.num_expand_context_chunk,
    )
    print(f"chatpdf model: {model}")
    # 调用 get_truth_dict 函数获取问题到答案的字典映射,并将结果合并到 truth_dict 中
    truth_dict = dict()
    for i in args.corpus_files.split(','):
        # 遍历每个jsonl
        if i.endswith('.jsonl'):
            tmp_truth_dict = get_truth_dict(i)
            truth_dict.update(tmp_truth_dict)
    print(f"truth_dict size: {len(truth_dict)}")
    # test data
    if args.query_file is None:
        examples = ["肛门病变可能是什么疾病的症状?", "膺窗穴的定位是什么?"]
    else:
        with open(args.query_file, 'r', encoding='utf-8') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    if args.test_size > 0:
        examples = examples[:args.test_size]
    print("Start inference.")
    # 批量生成和评估
    t1 = time.time()
    counts = 0
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    eval_batch_size = args.eval_batch_size
    # 将query划分批次，每个批次包含 eval_batch_size 个查询，tqdm的第一个参数是可迭代对象
    for batch in tqdm(
            [
                examples[i: i + eval_batch_size]
                for i in range(0, len(examples), eval_batch_size)
            ],
            desc="Generating outputs",
    ):
        results = []
        # 遍历查询,使用 ChatPDF模型生成回答,并将输入、生成的回答、参考结果和真实答案打印出来
        for example in batch:
            response, reference_results = model.predict(example)
            truth = truth_dict.get(example, '')
            print(f"===")
            print(f"Input: {example}")
            print(f"Reference: {reference_results}")
            print(f"Output: {response}")
            print(f"Truth: {truth}\n")
            results.append({"Input": example, "Output": response, "Truth": truth})
            counts += 1
        with open(args.output_file, 'a', encoding='utf-8') as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    t2 = time.time()
    print(f"Saved to {args.output_file}, Time cost: {t2 - t1:.2f}s, size: {counts}, "
          f"speed: {counts / (t2 - t1):.2f} examples/s")
