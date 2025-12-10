# hybrid_search.py
import re
from typing import List, Dict, Any, Optional, Union

import os
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Plus 
import faiss

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("russian")


CHUNKS_CSV_PATH = "websites_chunks.csv"
EMBED_MODEL_NAME = "ai-forever/FRIDA"


def simple_tokenize(text: str) -> List[str]:
    """Токенизация для BM25 с русским стеммингом."""
    tokens = re.findall(r"\w+", text.lower())
    return [stemmer.stem(t) for t in tokens]


class HybridSearchIndex:
    """
    Гибридный поиск:
      - BM25 по chunk_text
      - плотные эмбеддинги + Faiss (cosine / inner product)
      - объединение кандидатов и комбинированный скор
      - агрегация на уровень документа (web_id)
      - опциональный rerank поверх top-K документов
    """

    def __init__(
        self,
        embed_model_name: str = EMBED_MODEL_NAME,
        embed_model_kwargs: Optional[Dict] = None,
        rerank_model_name: Optional[str] = None,
        rerank_kwargs: Optional = None,
        cache_folder: str = "./models"
    ):
        self.embed_model_name = embed_model_name
        self.embed_model_kwargs = embed_model_kwargs if embed_model_kwargs else {}
        self.model: Optional[SentenceTransformer] = None
        self.cache_folder = cache_folder

        # --- CrossEncoder reranker ---
        self.rerank_model_name = rerank_model_name
        self.rerank_kwargs = rerank_kwargs
        self.reranker: Optional[CrossEncoder] = None

        self.chunks: List[Dict[str, Any]] = []   # [{web_id, chunk_id, text, url, title, kind}, ...]
        self.corpus_tokens: List[List[str]] = [] # для BM25

        # self.bm25: Optional[BM25Okapi] = None
        self.bm25: Optional[BM25Plus] = None

        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

    # ---------- инициализация моделей ----------

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(
                self.embed_model_name,
                trust_remote_code=True,
                model_kwargs=self.embed_model_kwargs,
                cache_folder=self.cache_folder,
            )

    def _load_reranker(self):
        """Ленивая загрузка CrossEncoder для rerank."""
        if self.reranker is None and self.rerank_model_name:
            self.reranker = CrossEncoder(
                self.rerank_model_name,
                trust_remote_code=True,
                model_kwargs=self.rerank_kwargs,
                cache_folder=self.cache_folder,
            )

    # ---------- построение индекса ----------

    def build_from_chunks_csv(self, path: str = CHUNKS_CSV_PATH, batch_size: int = 64):
        """
        Загружает websites_chunks.csv и строит:
          - BM25 корпус
          - эмбеддинги чанков (FRIDA)
          - Faiss-индекс
        """
        df = pd.read_csv(path)
        df = df.dropna(subset=["chunk_text"])

        self.chunks.clear()
        self.corpus_tokens.clear()

        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Load chunks"):
            text = getattr(row, "chunk_text")
            if not isinstance(text, str) or not text.strip():
                continue

            web_id  = getattr(row, "web_id")
            kind    = getattr(row, "kind", None)
            url     = getattr(row, "url", None)
            title   = getattr(row, "title", None)
            chunk_id = getattr(row, "chunk_id")

            meta = {
                "web_id": web_id,
                "kind": kind,
                "url": url,
                "title": title,
                "chunk_id": int(chunk_id),
                "text": text,
            }
            self.chunks.append(meta)
            self.corpus_tokens.append(simple_tokenize(text))

        # BM25
        # self.bm25 = BM25Okapi(self.corpus_tokens)
        self.bm25 = BM25Plus(self.corpus_tokens)

        # Эмбеддинги
        self._load_model()
        texts = [c["text"] for c in self.chunks]
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            prompt_name="document",
        ).astype("float32")

        self.embeddings = embs

        # Faiss по inner product (cosine при нормализации)
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)

        print(f"Built hybrid index for {len(self.chunks)} chunks")

        # ---------- сохранение / загрузка индекса на диск ----------

    def save_hybrid_index(self, dir_path: str) -> None:
        """
        Сохраняет Faiss-индекс, эмбеддинги и метаданные (chunks + corpus_tokens) на диск.
        """
        if (
            self.faiss_index is None
            or self.embeddings is None
            or self.bm25 is None
            or not self.chunks
        ):
            raise RuntimeError("Индекс ещё не построен, сохранять нечего.")

        os.makedirs(dir_path, exist_ok=True)

        # Faiss
        faiss_path = os.path.join(dir_path, "faiss.index")
        faiss.write_index(self.faiss_index, faiss_path)

        # Эмбеддинги
        emb_path = os.path.join(dir_path, "embeddings.npy")
        np.save(emb_path, self.embeddings)

        # Метаданные для BM25 и поиска
        meta = {
            "chunks": self.chunks,
            "corpus_tokens": self.corpus_tokens,
        }
        meta_path = os.path.join(dir_path, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load_hybrid_index(self, dir_path: str) -> None:
        """
        Загружает Faiss-индекс, эмбеддинги и BM25/метаданные с диска.
        """
        faiss_path = os.path.join(dir_path, "faiss.index")
        emb_path = os.path.join(dir_path, "embeddings.npy")
        meta_path = os.path.join(dir_path, "meta.json")

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"Не найден Faiss индекс: {faiss_path}")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Не найден файл эмбеддингов: {emb_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Не найден файл метаданных: {meta_path}")

        # Faiss
        self.faiss_index = faiss.read_index(faiss_path)

        # Эмбеддинги
        self.embeddings = np.load(emb_path)

        # Метаданные и BM25
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.chunks = meta["chunks"]
        self.corpus_tokens = meta["corpus_tokens"]
        self.bm25 = BM25Plus(self.corpus_tokens)

    # ---------- поиск ----------

    def search(
        self,
        query: str,
        top_k_dense: int = 400,
        top_k_bm25: int = 150,
        top_k_final_docs: int = 5,
        w_dense: float = 0.8,
        w_bm25: float = 0.2,
        multiquery: bool = False,
        doc_top_n_chunks: int = 3,   # сколько лучших чанков учитываем на документ
        rerank: bool = False,
        rerank_k: int = 20,          # сколько документов отдаём на rerank
    ) -> List[Dict[str, Any]]:
        if self.bm25 is None or self.faiss_index is None or self.embeddings is None:
            raise RuntimeError("Index is not built. Call build_from_chunks_csv(...) first.")

        # --- BM25 ---
        q_tokens = simple_tokenize(query[0]) if multiquery else simple_tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype="float32")
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:min(top_k_bm25, len(self.chunks))]

        # --- dense (FRIDA) ---
        self._load_model()
        q_vec = self.model.encode(
            query if multiquery else [query],
            convert_to_numpy=True,
            normalize_embeddings=False if multiquery else True,
            prompt_name="query",
        ).astype("float32")
        
        if multiquery: 
            q_vec = np.sum(q_vec, axis=0, keepdims=True) / len(query)
            q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12
            q_vec = q_vec / q_norm
            
        D, I = self.faiss_index.search(q_vec, min(top_k_dense, len(self.chunks)))
        top_dense_idx = I[0]
        dense_scores = D[0]

        dense_dict = {int(i): float(s) for i, s in zip(top_dense_idx, dense_scores)}

        # --- объединение кандидатов ---
        candidate_idx = sorted(set(top_bm25_idx.tolist()) | set(top_dense_idx.tolist()))
        if not candidate_idx:
            return []

        cand_bm25 = bm25_scores[candidate_idx]
        if cand_bm25.max() - cand_bm25.min() > 1e-6:
            bm25_norm_arr = (cand_bm25 - cand_bm25.min()) / (cand_bm25.max() - cand_bm25.min())
        else:
            bm25_norm_arr = np.zeros_like(cand_bm25)

        cand_dense = np.array([dense_dict.get(i, 0.0) for i in candidate_idx], dtype="float32")
        if cand_dense.max() - cand_dense.min() > 1e-6:
            dense_norm_arr = (cand_dense - cand_dense.min()) / (cand_dense.max() - cand_dense.min())
        else:
            dense_norm_arr = np.zeros_like(cand_dense)

        combined_scores = w_dense * dense_norm_arr + w_bm25 * bm25_norm_arr
        chunk_score = {idx: float(score) for idx, score in zip(candidate_idx, combined_scores)}

        # --- агрегация на уровень документа: top-N чанков ---
        doc_chunk_scores: Dict[Any, List[float]] = {}
        doc_best_chunk_meta: Dict[Any, Dict[str, Any]] = {}

        for idx in candidate_idx:
            meta = self.chunks[idx]
            doc_id = meta["web_id"]
            score = chunk_score[idx]

            doc_chunk_scores.setdefault(doc_id, []).append(score)

            if doc_id not in doc_best_chunk_meta or score > doc_best_chunk_meta[doc_id]["score"]:
                doc_best_chunk_meta[doc_id] = {
                    "web_id": meta["web_id"],
                    "score": score,
                    "url": meta["url"],
                    "title": meta["title"],
                    "kind": meta["kind"],
                    "chunk_id": meta["chunk_id"],
                    "chunk_text": meta["text"],
                }

        # итоговый скор документа = среднее по top-N чанкам
        doc_scores: Dict[Any, float] = {}
        for doc_id, scores in doc_chunk_scores.items():
            scores_sorted = sorted(scores, reverse=True)
            top_scores = scores_sorted[:doc_top_n_chunks]
            doc_scores[doc_id] = float(sum(top_scores) / len(top_scores))

        # сортируем все документы по doc_scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # сколько документов отдаём на rerank
        if rerank and self.reranker is not None and rerank_k > 0:
            candidate_doc_count = min(max(top_k_final_docs, rerank_k), len(sorted_docs))
        else:
            candidate_doc_count = min(top_k_final_docs, len(sorted_docs))

        candidate_docs = sorted_docs[:candidate_doc_count]

        # превращаем в список dict'ов (как раньше results)
        candidates: List[Dict[str, Any]] = []
        for doc_id, score in candidate_docs:
            info = doc_best_chunk_meta[doc_id]
            info = dict(info)
            info["score"] = score  # doc-level score
            candidates.append(info)

        # --- если rerank не нужен или не инициализирован --- 
        if not rerank or self.reranker is None:
            # просто возвращаем top_k_final_docs
            return candidates[:top_k_final_docs]

        # --- CrossEncoder.rank rerank ---
        self._load_reranker()
        if self.reranker is None:
            # на случай, если модель не загрузилась
            return candidates[:top_k_final_docs]

        main_query = query[0] if multiquery else query
        passages = [c["chunk_text"] for c in candidates]

        # сколько документов хотим получить после rerank
        top_k_rank = min(top_k_final_docs, len(passages))

        ranks = self.reranker.rank(
            main_query,
            passages,
            top_k=top_k_rank,
            return_documents=False,
            batch_size=32,
            show_progress_bar=False,
        )

        reranked: List[Dict[str, Any]] = []
        for item in ranks:
            corpus_id = int(item["corpus_id"])
            score = float(item["score"])
            c = dict(candidates[corpus_id])
            c["score"] = score
            reranked.append(c)

        return reranked
    

    def search_for_rag(
        self,
        query: Union[str, List[str]],
        top_k_dense: int = 400,
        top_k_bm25: int = 150,
        top_k_final_docs: int = 5,
        w_dense: float = 0.8,
        w_bm25: float = 0.2,
        multiquery: bool = False,
        doc_top_n_chunks: int = 3,
        max_chunks_per_doc: int = 3,
        max_chunks_total: Optional[int] = 10,
        rerank: bool = False,
        rerank_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Вариант search(...) для RAG: вместо одного лучшего чанка на документ
        возвращает несколько чанков на документ, сохраняя ранжирование документов.

        Возвращает список чанков (отсортирован по документам и их важности).
        Для каждого элемента словарь с полями:
            - web_id, kind, url, title, chunk_id, chunk_text
            - chunk_score: комбинированный скор (BM25 + dense) для этого чанка
            - doc_score: итоговый скор документа, к которому относится чанк
        """
        if self.bm25 is None or self.faiss_index is None or self.embeddings is None:
            raise RuntimeError("Index is not built. Call build_from_chunks_csv(...) first.")

        # --- BM25 по чанкам ---
        if multiquery:
            if not isinstance(query, list) or not query:
                raise ValueError("For multiquery=True expected non-empty List[str] as query.")
            q_tokens = simple_tokenize(query[0])
        else:
            if not isinstance(query, str):
                raise ValueError("For multiquery=False expected str as query.")
            q_tokens = simple_tokenize(query)

        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype="float32")
        top_bm25_idx = np.argsort(bm25_scores)[::-1][: min(top_k_bm25, len(self.chunks))]

        # --- dense (FRIDA) ---
        self._load_model()
        q_vec = self.model.encode(
            query if multiquery else [query],
            convert_to_numpy=True,
            normalize_embeddings=False if multiquery else True,
            prompt_name="query",
        ).astype("float32")

        if multiquery:
            # усредняем несколько запросов и нормализуем
            q_vec = np.sum(q_vec, axis=0, keepdims=True) / len(query)
            q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12
            q_vec = q_vec / q_norm

        D, I = self.faiss_index.search(q_vec, min(top_k_dense, len(self.chunks)))
        top_dense_idx = I[0]
        dense_scores = D[0]
        dense_dict = {int(i): float(s) for i, s in zip(top_dense_idx, dense_scores)}

        # --- объединяем кандидатов на уровне чанков ---
        candidate_idx = sorted(set(top_bm25_idx.tolist()) | set(top_dense_idx.tolist()))
        if not candidate_idx:
            return []

        cand_bm25 = bm25_scores[candidate_idx]
        if cand_bm25.max() - cand_bm25.min() > 1e-6:
            bm25_norm_arr = (cand_bm25 - cand_bm25.min()) / (cand_bm25.max() - cand_bm25.min())
        else:
            bm25_norm_arr = np.zeros_like(cand_bm25)

        cand_dense = np.array([dense_dict.get(i, 0.0) for i in candidate_idx], dtype="float32")
        if cand_dense.max() - cand_dense.min() > 1e-6:
            dense_norm_arr = (cand_dense - cand_dense.min()) / (cand_dense.max() - cand_dense.min())
        else:
            dense_norm_arr = np.zeros_like(cand_dense)

        combined_scores = w_dense * dense_norm_arr + w_bm25 * bm25_norm_arr
        chunk_score = {idx: float(score) for idx, score in zip(candidate_idx, combined_scores)}

        # --- агрегация чанков в документы ---
        # doc_chunks[doc_id] -> список (chunk_index, chunk_score)
        doc_chunks: Dict[Any, List[Any]] = {}
        for idx in candidate_idx:
            meta = self.chunks[idx]
            doc_id = meta["web_id"]
            score = chunk_score[idx]
            doc_chunks.setdefault(doc_id, []).append((idx, score))

        # итоговый скор документа = среднее по top-N чанкам
        doc_scores: Dict[Any, float] = {}
        for doc_id, lst in doc_chunks.items():
            scores_sorted = sorted((s for _, s in lst), reverse=True)
            if doc_top_n_chunks > 0:
                scores_sorted = scores_sorted[:doc_top_n_chunks]
            if not scores_sorted:
                continue
            doc_scores[doc_id] = float(sum(scores_sorted) / len(scores_sorted))

        if not doc_scores:
            return []

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # --- сколько документов пойдет на rerank ---
        if rerank:
            self._load_reranker()

        if rerank and self.reranker is not None and rerank_k > 0:
            candidate_doc_count = min(max(top_k_final_docs, rerank_k), len(sorted_docs))
        else:
            candidate_doc_count = min(top_k_final_docs, len(sorted_docs))

        candidate_docs = sorted_docs[:candidate_doc_count]
        if not candidate_docs:
            return []

        # --- опциональный CrossEncoder-rerank на уровне документов ---
        if not rerank or self.reranker is None or rerank_k <= 0:
            final_docs = candidate_docs[:top_k_final_docs]
        else:
            main_query = query[0] if multiquery else query

            # берём лучший чанк для каждого документа как "пассаже" для rerank
            passages: List[str] = []
            for doc_id, _ in candidate_docs:
                lst = doc_chunks[doc_id]
                best_idx, _ = max(lst, key=lambda x: x[1])
                passages.append(self.chunks[best_idx]["text"])

            top_k_rank = min(top_k_final_docs, len(passages))
            ranks = self.reranker.rank(
                main_query,
                passages,
                top_k=top_k_rank,
                return_documents=False,
                batch_size=32,
                show_progress_bar=False,
            )

            final_docs: List[Any] = []
            for item in ranks:
                corpus_id = int(item["corpus_id"])
                if corpus_id < 0 or corpus_id >= len(candidate_docs):
                    continue
                doc_id, _ = candidate_docs[corpus_id]
                score = float(item["score"])
                final_docs.append((doc_id, score))
        # если rerank не используется
        # if not rerank or self.reranker is None or rerank_k <= 0:
        #     final_docs = candidate_docs[:top_k_final_docs]

        # --- собираем чанки для RAG ---
        results: List[Dict[str, Any]] = []
        for doc_id, doc_score in final_docs:
            lst = doc_chunks.get(doc_id, [])
            if not lst:
                continue

            # сортируем чанки документа по их chunk_score
            lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
            for idx, ch_score in lst_sorted[:max_chunks_per_doc]:
                meta = self.chunks[idx]
                results.append(
                    {
                        "web_id": meta["web_id"],
                        "kind": meta.get("kind"),
                        "url": meta.get("url"),
                        "title": meta.get("title"),
                        "chunk_id": meta.get("chunk_id"),
                        "chunk_text": meta["text"],
                        "chunk_score": float(ch_score),
                        "doc_score": float(doc_score),
                    }
                )
                if max_chunks_total is not None and len(results) >= max_chunks_total:
                    break

            if max_chunks_total is not None and len(results) >= max_chunks_total:
                break

        return results



if __name__ == "__main__":
    # пример использования
    idx = HybridSearchIndex(
        embed_model_name=EMBED_MODEL_NAME,
        embed_model_kwargs={},  # при необходимости можно пробросить kwargs
        # rerank_model_name=RERANK_MODEL_NAME,
        # rerank_use_fp16=True,
    )
    idx.build_from_chunks_csv("websites_chunks.csv")

    q = "где посмотреть реквизиты счёта и БИК"
    results = idx.search(
        q,
        top_k_dense=400,
        top_k_bm25=150,
        top_k_final_docs=5,
        w_dense=0.9,
        w_bm25=0.1,
        doc_top_n_chunks=3,
        rerank=False,
    )

    for r in results:
        print("=" * 80)
        print(f"web_id={r['web_id']}  score={r['score']:.4f}")
        print(f"kind={r['kind']}  url={r['url']}")
        print(f"title={r['title']}")
        print(f"chunk_id={r['chunk_id']}")
        print(r["chunk_text"][:500], "...")
