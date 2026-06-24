from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
def register_reasoning_parser():
    from .hcx_reasoner import HcxReasoningParser
    ReasoningParserManager.register_module(name="hcx", module=HcxReasoningParser, force=True)

def register_tool_parser():
    from .hcx_tool_parser import HcxToolParser
    ToolParserManager.register_module(name="hcx", module=HcxToolParser, force=True)
