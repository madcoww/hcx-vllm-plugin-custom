# HyperCLOVAX vLLM Plugin
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

class HcxStreamingParserFunctionsMixin:
    # 사용하는 클래스에서 self.buffer_string / self.special_strings 를 초기화할 것

    def check_is_part_of_special_string(self):
        for ss in self.special_strings:
            min_len = min(len(self.buffer_string), len(ss))
            for ln in range(min_len, 0, -1):
                if self.buffer_string[-ln:] == ss[:ln]:
                    return True
        return False
