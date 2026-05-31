from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star


class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.pending_requests: Dict[str, Dict[str, Any]] = {}
        self.processed_requests: set[str] = set()
        self._parse_config(config)

        metadata = getattr(self, "metadata", None)
        version = metadata.version if metadata else "Unknown"
        logger.info(
            f"已加载 [KeywordApiFailover] 插件 v{version}，命中错误关键词后将改为调用指定 LLM 提供商返回结果。"
        )

    def _parse_config(self, config: AstrBotConfig) -> None:
        keywords_str = config.get(
            "error_keywords",
            "抱歉，我不能\nsorry\n作为ai\n不能帮助处理\n无法满足该请求\n对不起\n人工智能\n很抱歉\n我无法\n作为 AI\n角色扮演",
        )
        self.error_keywords = [
            item.strip().lower() for item in str(keywords_str).split("\n") if item.strip()
        ]

        self.provider_id = str(config.get("provider_id", "")).strip()
        self.forward_images = self._to_bool(config.get("forward_images", False))
        self.fallback_reply = str(
            config.get(
                "fallback_reply",
                "抱歉，备用模型暂时也不可用，请稍后再试。",
            )
        )

    def _to_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "启用", "是"}

    def _get_request_key(self, event: AstrMessageEvent) -> str:
        message_id = getattr(event.message_obj, "message_id", "")
        sender_id = event.get_sender_id()
        session_info = getattr(event, "unified_msg_origin", "")
        content = getattr(event, "message_str", "") or ""
        digest = hashlib.sha256(f"{sender_id}:{session_info}:{message_id}:{content}".encode("utf-8")).hexdigest()[:16]
        return f"keyword_api_failover:{sender_id}:{message_id}:{digest}"

    @filter.on_llm_request(priority=200)
    async def store_llm_request(self, event: AstrMessageEvent, req):
        if not hasattr(req, "prompt") or not hasattr(req, "contexts"):
            return

        request_key = self._get_request_key(event)
        system_prompt = getattr(req, "system_prompt", "") or ""
        if not system_prompt and hasattr(req, "conversation") and req.conversation:
            system_prompt = getattr(req.conversation, "system_prompt", "") or ""

        self.pending_requests[request_key] = {
            "prompt": getattr(req, "prompt", "") or "",
            "image_urls": copy.deepcopy(getattr(req, "image_urls", [])),
            "contexts": copy.deepcopy(getattr(req, "contexts", [])),
            "system_prompt": system_prompt,
        }

    def _find_error_keyword(self, text: str) -> Optional[str]:
        if not text:
            return None
        lowered = text.lower()
        for keyword in self.error_keywords:
            if keyword and keyword in lowered:
                return keyword
        return None

    def _strip_image_parts_from_content(self, content: Any) -> Any:
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"image_url", "image", "input_image"}:
                    continue
                new_parts.append(self._strip_image_parts_from_content(part))
            return new_parts
        if isinstance(content, dict):
            return {
                key: self._strip_image_parts_from_content(value)
                for key, value in content.items()
                if key not in {"image_url", "image", "input_image"}
            }
        return content

    def _prepare_contexts(self, contexts: Any) -> Any:
        contexts = copy.deepcopy(contexts)
        if self.forward_images:
            return contexts
        if not isinstance(contexts, list):
            return contexts
        for item in contexts:
            if isinstance(item, dict) and "content" in item:
                item["content"] = self._strip_image_parts_from_content(item.get("content"))
        return contexts

    async def _call_configured_provider(self, event: AstrMessageEvent, stored: Dict[str, Any]):
        provider_id = self.provider_id
        if not provider_id:
            provider_id = await self.context.get_current_chat_provider_id(
                umo=getattr(event, "unified_msg_origin", None)
            )
        if not provider_id:
            raise RuntimeError("未选择备用 LLM 提供商，且当前会话没有可用提供商")

        image_urls = copy.deepcopy(stored.get("image_urls", [])) if self.forward_images else None
        contexts = self._prepare_contexts(stored.get("contexts", []))

        return await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=stored.get("prompt", "") or None,
            image_urls=image_urls or None,
            system_prompt=stored.get("system_prompt", "") or None,
            contexts=contexts or None,
        )

    def _extract_llm_response_text(self, resp: Any) -> str:
        text = getattr(resp, "completion_text", "") or ""
        if text:
            return str(text).strip()
        result_chain = getattr(resp, "result_chain", None)
        if result_chain and hasattr(result_chain, "get_plain_text"):
            return str(result_chain.get_plain_text() or "").strip()
        return ""

    def _set_text_result(self, event: AstrMessageEvent, text: str) -> None:
        event.set_result(event.plain_result(text))

    async def _handle_keyword_hit(self, event: AstrMessageEvent, request_key: str, text: str) -> bool:
        keyword = self._find_error_keyword(text)
        if not keyword:
            return False

        if request_key in self.processed_requests:
            return False
        self.processed_requests.add(request_key)

        logger.warning(f"检测到错误关键词 '{keyword}'，放弃原回复并调用备用 LLM 提供商。")
        stored = self.pending_requests.get(request_key, {"prompt": event.message_str or "", "contexts": [], "image_urls": []})

        try:
            llm_resp = await self._call_configured_provider(event, stored)
            new_text = self._extract_llm_response_text(llm_resp)
            if not new_text:
                raise RuntimeError("备用 LLM 提供商返回成功，但未提取到文本内容")
            self._set_text_result(event, new_text)
            logger.info("备用 LLM 提供商调用成功，已替换原始回复。")
            return True
        except Exception as exc:
            logger.error(f"调用备用 LLM 提供商失败: {exc}", exc_info=True)
            if self.fallback_reply.strip():
                self._set_text_result(event, self.fallback_reply.strip())
                return True
            return False
        finally:
            self.pending_requests.pop(request_key, None)

    @filter.on_llm_response(priority=10)
    async def retry_on_llm_response(self, event: AstrMessageEvent, resp):
        request_key = self._get_request_key(event)
        text = getattr(resp, "completion_text", "") or ""
        if text:
            await self._handle_keyword_hit(event, request_key, text)

    @filter.on_decorating_result(priority=-100)
    async def check_and_replace(self, event: AstrMessageEvent, *args, **kwargs):
        request_key = self._get_request_key(event)
        if request_key in self.processed_requests:
            self.processed_requests.discard(request_key)
            self.pending_requests.pop(request_key, None)
            return

        result = event.get_result()
        if not result or not hasattr(result, "get_plain_text"):
            self.pending_requests.pop(request_key, None)
            return

        text = result.get_plain_text() or ""
        handled = await self._handle_keyword_hit(event, request_key, text)
        if not handled:
            self.pending_requests.pop(request_key, None)

    async def terminate(self):
        self.pending_requests.clear()
        self.processed_requests.clear()
        logger.info("已卸载 [KeywordApiFailover] 插件并清理缓存。")