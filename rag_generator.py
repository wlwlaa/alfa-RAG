# rag_generator.py
from typing import Any, Dict, List, Optional
import textwrap

from openai import OpenAI  # официальный Python SDK


class RAGAnswerGenerator:
    """
    Генеративный модуль поверх HybridSearchIndex.

    Предполагается, что LLM развёрнута через vLLM OpenAI-compatible server
    и доступна по base_url вида "http://localhost:8000/v1".

    Мы используем официальный Python-клиент OpenAI, но ходим не в облако,
    а в локальный vLLM (инференс локальный, что соответствует ограничениям кейса).
    """

    def __init__(
        self,
        index,
        model_name: str,
        api_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        openai_client: Optional[OpenAI] = None,
        system_prompt: Optional[str] = None,
        max_context_chars: int = 3500,
        max_answer_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        timeout: int = 30,
        extra_body: Dict = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False},},
    ) -> None:
        """
        :param index: экземпляр HybridSearchIndex
        :param model_name: имя модели, как она зарегистрирована в vLLM
        :param api_url: base_url для OpenAI-клиента (обычно "http://localhost:8000/v1")
        :param api_key: формальный ключ (для vLLM можно оставить "EMPTY")
        :param openai_client: опционально — уже созданный OpenAI(client),
                              если хочешь переиспользовать его снаружи
        """
        self.index = index
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.max_context_chars = max_context_chars
        self.max_answer_tokens = max_answer_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.extra_body = extra_body

        # Инициализируем OpenAI-клиент.
        # Если снаружи передали готовый client — используем его.
        if openai_client is not None:
            self.client = openai_client
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.api_url,
            )

        self.system_prompt = system_prompt or (
            "Ты — виртуальный помощник Альфа-Банка.\n"
            "Отвечай по-русски, вежливо и по делу.\n"
            "Опирайся только на предоставленный контекст и не придумывай факты, "
            "которых в нём нет. Если информации недостаточно — честно скажи об этом.\n"
            "Постарайся ответить в 2–5 предложениях без лишней воды."
        )

    # ---------- публичный интерфейс ----------

    def answer_question(
        self,
        question: str | List[str],
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Запускает retrieval -> grounding -> generation для одного вопроса.

        Возвращает словарь:
            {
                "answer": str,
                "used_chunks": List[Dict[str, Any]],
            }
        """
        search_kwargs = search_kwargs or {}

        # 1. retrieval: достаём несколько чанков на документ
        chunks = self.index.search_for_rag(question, **search_kwargs)

        if type(question) == list:
            question = question[0]

        # 2. нет релевантного контекста
        if not chunks:
            return {
                "answer": self._fallback_no_context(question),
                "used_chunks": [],
            }

        # 3. grounding: собираем текстовый контекст
        context = self._build_context(chunks)

        # 4. generation: вызываем LLM через OpenAI SDK -> vLLM
        messages = self._build_messages(question, context)
        answer_text = self._call_llm(messages)

        return {
            "answer": answer_text,
            "used_chunks": chunks,
        }

    # ---------- внутренние методы ----------

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Формирует текст контекста в виде блоков [DOC i].
        Пытается не выходить за пределы max_context_chars.
        """
        blocks: List[str] = []
        total_len = 0

        for i, ch in enumerate(chunks, start=1):
            header_lines = [f"[DOC {i}] web_id={ch.get('web_id')}"]
            if ch.get("kind"):
                header_lines.append(f"Тип: {ch['kind']}")
            if ch.get("title"):
                header_lines.append(f"Заголовок: {ch['title']}")
            if ch.get("url"):
                header_lines.append(f"URL: {ch['url']}")

            header = "\n".join(header_lines)
            # text = (ch.get("chunk_text") or "").strip()
            text = self._clean_chunk_text(ch)
            block = f"{header}\nТекст:\n{text}\n"

            new_total = total_len + len(block)
            if new_total > self.max_context_chars and blocks:
                break

            blocks.append(block)
            total_len = new_total

        return "\n\n".join(blocks)

    def _build_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        """
        Подготавливает messages для chat-completions.
        """
        user_prompt = textwrap.dedent(
            f"""
            Вопрос клиента:
            {question}

            Контекст (фрагменты документов):
            {context}

            Сформулируй ответ клиенту, используя только информацию из контекста.
            Ответь в 2–5 предложениях, без лишних отступлений.
            Если нужной информации в контексте нет или её недостаточно для точного ответа,
            честно напиши об этом.
            """
        ).strip()

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Вызов локального LLM через OpenAI Python SDK (клиент),
        который ходит в OpenAI-совместимый сервер vLLM.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_answer_tokens,
                timeout=self.timeout,
                extra_body=self.extra_body,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            # На проде можно логировать e; здесь — безопасный fallback
            print(e)
            return (
                "К сожалению, не удалось сформировать ответ из-за внутренней ошибки модели. "
                "Попробуйте повторить запрос позже."
            )

    def _clean_chunk_text(self, chunk: dict) -> str:
        """
        Убирает из начала чанка дублирующий title + перенос строки.
        Ожидаемый формат: "<title>\\n\\nостальной текст".
        Работает толерантно к неразрывным пробелам и лишним пробелам.
        """
        raw_text = (chunk.get("chunk_text") or "").lstrip()
        title = str((chunk.get("title") or "")).strip()
        if not raw_text or not title:
            return raw_text

        # нормализация пробелов и регистра для сравнения
        def norm(s: str) -> str:
            s = s.replace("\xa0", " ")  # неразрывный пробел -> обычный
            return " ".join(s.split()).strip().lower()

        text = raw_text
        title_norm = norm(title)

        # 1) точный префикс "title\n\n" или "title\n"
        prefix1 = title + "\n\n"
        prefix2 = title + "\n"
        if text.startswith(prefix1):
            return text[len(prefix1):].lstrip()
        if text.startswith(prefix2):
            return text[len(prefix2):].lstrip()

        # 2) хак: первая "порция" до двойного перевода строки = title (с учётом нормализации)
        first_block, *rest = text.split("\n\n", 1)
        if norm(first_block) == title_norm and rest:
            return rest[0].lstrip()

        # 3) крайний случай: первая строка почти точно такой же title — убираем её
        first_line, *rest_lines = text.split("\n", 1)
        if norm(first_line) == title_norm and rest_lines:
            return rest_lines[0].lstrip()

        # если ничего не сработало — возвращаем как есть
        return text

    def _fallback_no_context(self) -> str:
        """
        Ответ по умолчанию, когда ретривер ничего не нашёл или контекст пустой.
        """
        return (
            "Нет эталонного ответа"
        )
