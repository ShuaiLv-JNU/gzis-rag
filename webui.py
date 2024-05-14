# -*- coding: utf-8 -*-
"""
webui.py
"""
import argparse
import os

import gradio as gr
from loguru import logger

from chatpdf import ChatPDF

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model_name", type=str, default="01-ai/Yi-6B-Chat")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--rerank_model_name", type=str, default="maidalun1020/bce-reranker-base_v1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--corpus_files", type=str, default="sample.pdf")
    parser.add_argument("--int4", action='store_true', help="use int4 quantization")
    parser.add_argument("--int8", action='store_true', help="use int8 quantization")
    parser.add_argument("--chunk_size", type=int, default=220)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--num_expand_context_chunk", type=int, default=1)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8082)
    parser.add_argument("--share", action='store_true', help="share model")
    args = parser.parse_args()
    logger.info(args)

    model = ChatPDF(
        generate_model_type=args.gen_model_type,
        generate_model_name_or_path=args.gen_model_name,
        lora_model_name_or_path=args.lora_model,
        corpus_files=args.corpus_files.split(','),
        device=args.device,
        int4=args.int4,
        int8=args.int8,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_expand_context_chunk=args.num_expand_context_chunk,
        rerank_model_name_or_path=args.rerank_model_name,
    )
    logger.info(f"chatpdf model: {model}")

    # 聊天机器人的流式预测接口,调用ChatPDF的predict_stream函数生成回答
    def predict_stream(message, history):
        history_format = []
        # **有点问题，history对于流式来说其实是空的** #
        for human, assistant in history:
            history_format.append([human, assistant])
        model.history = history_format
        # **有点问题** #
        for chunk in model.predict_stream(message):
            yield chunk

    # 聊天机器人的非流式预测接口,调用ChatPDF的predict函数生成回答
    def predict(message, history):
        logger.debug(message)
        response, reference_results = model.predict(message)
        r = response + "\n\n" + '\n'.join(reference_results)
        logger.debug(r)
        return r

    # Gradio的Chatbot组件,用于显示聊天记录
    chatbot_stream = gr.Chatbot(
        height=600,
        avatar_images=(
            os.path.join(pwd_path, "assets/user.png"),
            os.path.join(pwd_path, "assets/llama.png"),
        ), bubble_full_width=False)

    title = " 🎉ChatPDF WebUI🎉 "
    description = "Link in Github: [ShuaiLv-JNU/gzis-rag](https://github.com/ShuaiLv-JNU/gzis-rag)"
    css = """.toast-wrap { display: none !important } """
    examples = ['Can you tell me about the NLP?', '介绍下NLP']
    # Gradio的ChatInterface组件,它将前面定义的各种元素组合成一个完整的聊天界面
    chat_interface_stream = gr.ChatInterface(
        predict_stream,
        textbox=gr.Textbox(lines=4, placeholder="Ask me question", scale=7),
        title=title,
        description=description,
        chatbot=chatbot_stream,
        css=css,
        examples=examples,
        theme='soft',
    )
    # Gradio的Blocks布局中,并调用launch()方法来启动Web服务
    with gr.Blocks() as demo:
        chat_interface_stream.render()
    demo.queue().launch(
        server_name=args.server_name, server_port=args.server_port, share=args.share
    )
