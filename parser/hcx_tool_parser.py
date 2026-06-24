# HyperCLOVAX vLLM Plugin
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import json
import re
from collections.abc import Sequence
from typing import Any, Union

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.logger import init_logger
from .hcx_parser_mixin import HcxStreamingParserFunctionsMixin

logger = init_logger(__name__)


class HcxToolParser(ToolParser, HcxStreamingParserFunctionsMixin):
    """HCX SEED-Think 챗 템플릿이 내보내는 tool-call 포맷을 파싱한다(14B/32B 공통):

        <tool_call>{함수명}
        <arg_key>{키}</arg_key>
        <arg_value>{값}</arg_value>
        ...
        </tool_call>

    인자가 문자열 하나인 경우의 <arguments>{json}</arguments> 형태도 허용한다.
    <think> 추론부는 HcxReasoningParser가 담당하므로, 이 파서는 추론이 끝난 뒤의
    content만 받는다(vLLM이 is_reasoning_end 이후에만 tool 파서를 호출함).
    """

    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.arguments_regex = re.compile(r"<arguments>(.*?)</arguments>", re.DOTALL)
        self.arg_pair_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

        # 스트리밍 상태 (vLLM이 요청마다 파서를 새로 생성한다)
        self.current_tool_id = -1
        self.buffer_string = ''
        # "<tool_call>"가 델타 경계에서 잘렸을 때 content로 흘리지 않도록 보류
        self.special_strings = [self.tool_call_start_token]

    @staticmethod
    def _coerce(value: str) -> Any:
        # 템플릿은 문자열 값은 원문 그대로, 그 외(숫자/불리언/객체)는 tojson으로 내보낸다.
        # 따라서 json 파싱을 시도하고, 실패하면 원문 문자열로 둔다.
        value = value.strip()
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    def _parse_tool_call(self, block: str) -> tuple[str, str]:
        """<tool_call>..</tool_call> 사이 블록 -> (함수명, arguments json 문자열)."""
        name, _, body = block.partition('\n')
        name = name.strip()

        # <arguments> 형태면 그 안의 json 문자열을 그대로 사용
        args_match = self.arguments_regex.search(body)
        if args_match:
            return name, args_match.group(1).strip()

        # arg_key/arg_value 쌍을 순서대로 dict로 복원
        obj = {k.strip(): self._coerce(v)
               for k, v in self.arg_pair_regex.findall(body)}
        return name, json.dumps(obj, ensure_ascii=False)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)
        try:
            tool_calls = []
            for block in self.tool_call_regex.findall(model_output):
                name, arguments = self._parse_tool_call(block)
                tool_calls.append(ToolCall(
                    type="function",
                    function=FunctionCall(name=name, arguments=arguments)))

            # 첫 <tool_call> 앞의 텍스트는 일반 content로 처리
            content = model_output[:model_output.find(
                self.tool_call_start_token)].rstrip('\n')
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content or None)
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        if self.tool_call_start_token in current_text:
            # 지금까지 '완결된'(</tool_call>까지 닫힌) 블록만 대상으로 한다
            blocks = self.tool_call_regex.findall(current_text)
            pending = blocks[self.current_tool_id + 1:]
            if not pending:
                return None

            # 이번 델타에서 새로 완결된 블록들을 한 번에 emit
            deltas = []
            for block in pending:
                self.current_tool_id += 1
                name, arguments = self._parse_tool_call(block)
                deltas.append(DeltaToolCall(
                    index=self.current_tool_id,
                    type="function",
                    id=f'hcx_tool_call_{self.current_tool_id}',
                    function=DeltaFunctionCall(
                        name=name, arguments=arguments).model_dump(
                            exclude_none=True)))
            return DeltaMessage(tool_calls=deltas)

        # 아직 tool_call 없음: content로 흘리되, 잘린 "<tool_call>" 꼬리는 보류
        self.buffer_string += delta_text
        if self.check_is_part_of_special_string():
            return None
        out = self.buffer_string
        self.buffer_string = ''
        return DeltaMessage(content=out) if out else None
