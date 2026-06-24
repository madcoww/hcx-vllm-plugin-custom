# HyperCLOVAX vLLM Plugin
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

from collections.abc import Sequence
from typing import Optional, Union

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)

from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
)

from vllm.reasoning import ReasoningParser

from .hcx_parser_mixin import HcxStreamingParserFunctionsMixin


class HcxReasoningParser(ReasoningParser, HcxStreamingParserFunctionsMixin):
    """Reasoning parser for the HCX `<think> ... </think>` chat format.

    The chat template injects `<think>` into the generation prompt, so the
    model output usually *starts* with the reasoning text and is closed by
    `</think>`; the answer follows. Parsing is therefore a single partition
    on `</think>`.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        # for structured-output helpers (is_reasoning_end / extract_content_ids)
        self.think_end_token_ids = tokenizer.encode(
            self.think_end_token, add_special_tokens=False)

        # streaming state (per-request; reset on stream start)
        self.buffer_string = ''
        self.reasoning_ended = False
        # mixin uses special_strings for end-of-buffer partial matching
        self.special_strings = [self.think_end_token, self.think_start_token]

    def extract_reasoning(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        end = self.think_end_token
        if end in model_output:
            reasoning, _, content = model_output.partition(end)
            reasoning = reasoning.replace(self.think_start_token, '').strip('\n')
            return (reasoning or None), (content.lstrip('\n') or None)

        # </think> never appeared:
        kwargs = request.chat_template_kwargs or {}
        # 템플릿 기본값이 thinking ON(미지정=ON)이므로 폴백도 True로 맞춘다
        if kwargs.get('enable_thinking', True):
            # thinking on but not closed (e.g. truncated) -> all reasoning
            reasoning = model_output.replace(self.think_start_token, '').strip('\n')
            return (reasoning or None), None
        # thinking off -> all content (template renders an empty <think></think>)
        return None, (model_output.lstrip('\n') or None)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        if not previous_text:
            # new stream -> reset state (in case the instance is reused)
            self.buffer_string = ''
            self.reasoning_ended = False

        if self.reasoning_ended:
            return DeltaMessage(content=delta_text)

        self.buffer_string += delta_text

        end = self.think_end_token
        if end in self.buffer_string:
            before, _, after = self.buffer_string.partition(end)
            self.buffer_string = ''
            self.reasoning_ended = True
            before = before.replace(self.think_start_token, '')
            msg = {}
            if before:
                msg['reasoning_content'] = before
            if after:
                msg['content'] = after
            return DeltaMessage(**msg) if msg else None

        # </think> may be split across deltas: hold the tail until it resolves
        if self.check_is_part_of_special_string():
            return None

        out = self.buffer_string.replace(self.think_start_token, '')
        self.buffer_string = ''
        # ponytail: thinking-off + streaming has no per-stream request here, so
        # output before any </think> is emitted as reasoning_content. Correct for
        # the common thinking-on case; harmless for thinking-off (no </think> ever).
        return DeltaMessage(reasoning_content=out) if out else None

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # </think> token sequence present? (structured output; best-effort)
        ids = self.think_end_token_ids
        n = len(ids)
        if n == 0:
            return True
        return any(input_ids[i:i + n] == ids
                   for i in range(len(input_ids) - n + 1))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        ids = self.think_end_token_ids
        n = len(ids)
        for i in range(len(input_ids) - n + 1):
            if input_ids[i:i + n] == ids:
                return input_ids[i + n:]
        return []
