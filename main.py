import torch
import argparse

import gradio as gr

from hybrid_search1 import HybridSearchIndex
from rag_generator import RAGAnswerGenerator

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device for inference: {device}")


MODEL_NAME = "models/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
HOST = "localhost"
PORT = "8080"

def init(args):
    idx = HybridSearchIndex(
        embed_model_name="sergeyzh/BERTA",
        cache_folder="./models"
    )

    try:
        idx.load_hybrid_index("./data/hybrid_index")
        print("Hybrid index loaded")
    except:
        idx.build_from_chunks_csv(path="./data/chunks_golden_v4_targ800_maxsize_1100_overlap20.csv", batch_size=32)
        idx.save_hybrid_index("./data/hybrid_index")
        print("Hybrid index built")

    rag = RAGAnswerGenerator(
        index=idx,
        model_name=args.model,
        api_url=f"http://{args.host}:{args.port}/v1",
        api_key="EMPTY",                     
        max_context_chars=8024,
        max_answer_tokens=380,
        system_prompt="""
    Ты — виртуальный помощник Альфа-Банка. Отвечай по-русски, вежливо и по делу.

    Правила работы:
    Отвечай только на основе предоставленного контекста.
    Не выдумывай проценты, суммы, сроки и другие цифры, которых нет в тексте.

    Если каких‑то деталей нет в контексте, прямо напиши, что
    этих деталей не хватает, но обязательно перечисли всё,
    что по вопросу есть в контексте.

    Если формулировка из контекста хорошо подходит к ответу,
    можно использовать её почти дословно.
    Если ответ естественно получается в виде списка шагов или причин,
    оформи его нумерованным списком, как в примерах.

    Пиши по‑делу: без длинных вступлений, оправданий и лишних эмоций.
    Старайся уложиться в 2–5 предложений. Ответ должен быть не сильно длиннее эталонных ответов и без «воды».
    """
    )

    return rag


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot (Gradio)")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Название модели"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=HOST,
        help="Хост модели LLM"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help="Порт модели LLM"
    )

    args = parser.parse_args()
    rag = init(args)

    def chat_fn(question, history):
        answer = rag.answer_question(
            question,
            search_kwargs={
                "top_k_final_docs": 4,
                "max_chunks_per_doc": 3,
                "max_chunks_total": 8,
                "w_dense": 0.7,
                "w_bm25": 0.3,
                "multiquery": False,
                "rerank": False,
            },
        )
        return answer['answer']

    gr.ChatInterface(
        fn=chat_fn,
        title="Alfa Chatbot",
        description="RAG pipline for Alfa Bank chat-bot question answering",
        concurrency_limit="default",
        api_name="chat"
    ).launch()


if __name__ == "__main__":
    main()