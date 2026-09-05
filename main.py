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
        self.replaced_texts: Dict[str, str] = {}
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

        # 轮换顺序：先用下拉选的 provider_id，再依次用文本框里的 provider_ids
        single = str(config.get("provider_id", "")).strip()
        provider_ids_str = str(config.get("provider_ids", "")).strip()
        extra = [p.strip() for p in provider_ids_str.split("\n") if p.strip()]
        # 合并去重，保持顺序
        seen: set = set()
        merged: list = []
        for p in ([single] if single else []) + extra:
            if p not in seen:
                seen.add(p)
                merged.append(p)
        # 都为空则运行时自动取当前会话 provider
        self.provider_ids = merged if merged else [None]

        self.provider_id = single
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

    def _list_providers(self) -> list:
        """返回 (provider_id, model_name) 列表。"""
        result = []
        try:
            for prov in self.context.get_all_providers() or []:
                pid = ""
                model = ""
                try:
                    meta = prov.meta()
                    pid = str(getattr(meta, "id", "") or "")
                    model = str(getattr(meta, "model", "") or "")
                except Exception:
                    pass
                if not pid:
                    pid = str(getattr(prov, "provider_config", {}).get("id", "") or "")
                if not model and hasattr(prov, "get_model"):
                    try:
                        model = str(prov.get_model() or "")
                    except Exception:
                        pass
                if pid:
                    result.append((pid, model))
        except Exception as exc:
            logger.warning(f"枚举 LLM 提供商失败: {exc}")
        return result

    def _resolve_provider_id(self, raw: str) -> str:
        """把用户填写的值解析成真实 Provider ID。支持直接填 Provider ID 或模型名。"""
        raw = str(raw).strip()
        if not raw:
            return raw
        providers = self._list_providers()
        if not providers:
            return raw

        ids = [pid for pid, _ in providers]
        # 1. 精确匹配 Provider ID
        if raw in ids:
            return raw
        lowered = raw.lower()
        # 2. 忽略大小写匹配 Provider ID
        for pid in ids:
            if pid.lower() == lowered:
                return pid
        # 3. 按模型名匹配
        for pid, model in providers:
            if model and model.lower() == lowered:
                logger.info(f"'{raw}' 是模型名，已解析为 Provider ID '{pid}'。")
                return pid
        # 4. 匹配 "id/model" 形式的后半段
        for pid, _ in providers:
            if "/" in pid and pid.split("/")[-1].lower() == lowered:
                logger.info(f"'{raw}' 已模糊匹配到 Provider ID '{pid}'。")
                return pid
        return raw

    async def _call_provider(self, event: AstrMessageEvent, stored: Dict[str, Any], provider_id: Optional[str]):
        if provider_id:
            provider_id = self._resolve_provider_id(provider_id)
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

    def _apply_replacement(
        self,
        event: AstrMessageEvent,
        request_key: str,
        text: str,
        resp: Any = None,
    ) -> None:
        """写回替换后的文本。

        在 on_llm_response 阶段必须直接改 resp 对象：AstrBot 在钩子返回后会用
        resp 重新构建 MessageChain 并 set_result，此时任何 event.set_result
        都会被覆盖。

        注意：completion_text 是 property，当 resp.result_chain 存在时 setter 会把
        文本写进 result_chain（先剔除所有 Plain 再插入新的），getter 也从 result_chain
        取值。因此绝不能在赋值后再清空 result_chain，否则 getter 会回退到空的
        _completion_text，替换结果丢失。

        改 resp 只影响展示。会话历史存的是 run_context.messages 里的 TextPart 快照，
        那份快照在触发本钩子之前就已建立，必须在 on_agent_done 里单独改写，
        所以这里把新文本记下来交给 rewrite_history。
        """
        self.replaced_texts[request_key] = text
        if resp is not None:
            try:
                resp.completion_text = text
            except Exception as exc:
                logger.warning(f"直接改写 LLMResponse 失败，回退到 set_result: {exc}")
        self._set_text_result(event, text)

    async def _handle_keyword_hit(
        self,
        event: AstrMessageEvent,
        request_key: str,
        text: str,
        resp: Any = None,
    ) -> bool:
        keyword = self._find_error_keyword(text)
        if not keyword:
            return False

        if request_key in self.processed_requests:
            return False
        self.processed_requests.add(request_key)

        logger.warning(f"检测到错误关键词 '{keyword}'，开始轮换备用 LLM 提供商。")
        stored = self.pending_requests.get(request_key, {"prompt": event.message_str or "", "contexts": [], "image_urls": []})

        # 构建轮换列表：有配置则用列表，否则尝试当前会话 provider（传 None 让 _call_provider 自动获取）
        providers_to_try: list = list(self.provider_ids) if self.provider_ids else [None]

        try:
            for provider_id in providers_to_try:
                label = provider_id or "(当前会话 provider)"
                try:
                    llm_resp = await self._call_provider(event, stored, provider_id)
                    new_text = self._extract_llm_response_text(llm_resp)
                    if not new_text:
                        logger.warning(f"备用提供商 {label} 返回成功但无文本，尝试下一个。")
                        continue
                    hit = self._find_error_keyword(new_text)
                    if hit:
                        logger.warning(f"备用提供商 {label} 的回复仍含关键词 '{hit}'，尝试下一个。")
                        continue
                    self._apply_replacement(event, request_key, new_text, resp)
                    logger.info(f"备用提供商 {label} 调用成功，已替换原始回复。")
                    return True
                except Exception as exc:
                    logger.error(f"调用备用提供商 {label} 失败: {exc}", exc_info=True)
                    if "not found" in str(exc).lower():
                        available = self._list_providers()
                        if available:
                            hint = "，".join(
                                f"{pid}（模型 {model or '未知'}）" for pid, model in available
                            )
                            logger.error(f"当前可用的 Provider ID：{hint}")
                    continue

            # 全部耗尽
            logger.error("所有备用 LLM 提供商均失败或仍含关键词，使用兜底回复。")
            if self.fallback_reply.strip():
                self._apply_replacement(
                    event, request_key, self.fallback_reply.strip(), resp
                )
                return True
            return False
        finally:
            self.pending_requests.pop(request_key, None)

    @filter.on_llm_response(priority=10)
    async def retry_on_llm_response(self, event: AstrMessageEvent, resp):
        request_key = self._get_request_key(event)
        text = getattr(resp, "completion_text", "") or ""
        if not text:
            result_chain = getattr(resp, "result_chain", None)
            if result_chain and hasattr(result_chain, "get_plain_text"):
                text = result_chain.get_plain_text() or ""
        if text:
            await self._handle_keyword_hit(event, request_key, text, resp=resp)

    @filter.on_agent_done()
    async def rewrite_history(self, event: AstrMessageEvent, run_context, response):
        """把替换后的文本写进会话历史。

        存进数据库的不是 LLMResponse，而是 run_context.messages 里那条 assistant
        消息的 TextPart —— 它在 on_llm_response 触发之前就已经用原始文本建好了，
        改 resp.completion_text 追不回去。on_agent_done 紧接在 on_llm_response
        之后触发，且仍早于 _save_to_history，是唯一能改到历史的时机。
        不修的话，下一轮对话模型会在上下文里看到自己上一轮的拒答文本。
        """
        request_key = self._get_request_key(event)
        new_text = self.replaced_texts.pop(request_key, None)
        if not new_text:
            return

        messages = getattr(run_context, "messages", None) or []
        for message in reversed(messages):
            if getattr(message, "role", "") != "assistant":
                continue
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                logger.warning("assistant 消息的 content 不是分片列表，无法改写历史。")
                return
            text_parts = [p for p in content if getattr(p, "type", None) == "text"]
            if not text_parts:
                logger.warning("assistant 消息中没有文本分片，无法改写历史。")
                return
            text_parts[0].text = new_text
            for extra in text_parts[1:]:
                content.remove(extra)
            logger.info("已将备用模型的回复同步写入会话历史。")
            return

    @filter.on_decorating_result(priority=-100)
    async def check_and_replace(self, event: AstrMessageEvent, *args, **kwargs):
        request_key = self._get_request_key(event)
        if request_key in self.processed_requests:
            self.processed_requests.discard(request_key)
            self.pending_requests.pop(request_key, None)
            self.replaced_texts.pop(request_key, None)
            return

        result = event.get_result()
        if not result or not hasattr(result, "get_plain_text"):
            self.pending_requests.pop(request_key, None)
            return

        text = result.get_plain_text() or ""
        handled = await self._handle_keyword_hit(event, request_key, text)
        if not handled:
            self.pending_requests.pop(request_key, None)
        # 本阶段已晚于 on_agent_done，历史快照改不到了，丢掉记录避免滞留
        if self.replaced_texts.pop(request_key, None):
            logger.warning(
                "关键词是在 on_decorating_result 阶段才命中的，已晚于历史保存时机，"
                "本次替换只影响发出去的内容。若该回复来自 LLM，会话历史里仍是原始回复。"
            )

    async def terminate(self):
        self.pending_requests.clear()
        self.processed_requests.clear()
        self.replaced_texts.clear()
        logger.info("已卸载 [KeywordApiFailover] 插件并清理缓存。")