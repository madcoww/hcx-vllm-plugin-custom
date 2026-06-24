# HyperCLOVAX vLLM Plugin

HyperCLOVAX(HCX) 모델을 vLLM에서 서빙하기 위한 플러그인. LLaMA 구조에
아래 변경을 적용한다.
- [μP](https://arxiv.org/pdf/2203.03466)
- [Peri-LN](https://arxiv.org/pdf/2502.02732)

> **기준 버전:** vLLM **v0.20.0** (`--reasoning-parser` / `--tool-call-parser`
> 인터페이스 및 `vllm.tool_parsers`, `vllm.entrypoints.openai.engine.protocol`
> 경로 기준). 그 이하 버전은 모듈 경로가 달라 동작하지 않을 수 있다.
>
> **지원 모델:** **HyperCLOVAX-SEED-Think-32B 전용.** 14B는 tool calling을 신뢰성 있게
> 수행하지 못해(추론 후 `<tool_call>` 미출력, temperature 0에서도 재현) **미지원**.

## 구성 요소

| 구성 | 파일 | 등록 이름 |
|------|------|-----------|
| 모델 | [model/vllm_hyperclovax.py](model/vllm_hyperclovax.py) | `HyperCLOVAXForCausalLM` |
| Reasoning 파서 | [parser/hcx_reasoner.py](parser/hcx_reasoner.py) | `hcx` |
| Tool 파서 | [parser/hcx_tool_parser.py](parser/hcx_tool_parser.py) | `hcx` |
| 챗 템플릿 | [chat_template_hcx.jinja](chat_template_hcx.jinja) | — |

`setup.py`의 `vllm.general_plugins` 엔트리포인트로 모델/파서가 자동 등록된다.
**챗 템플릿은 자동 주입되지 않으므로** 기동 시 `--chat-template`로 지정해야 한다.

## 모델 설정 (μP / Peri-LN)

[configuration_hyperclovax.py](model/configuration_hyperclovax.py)

- μP
  - **embedding_multiplier** (`float`, default `None`) — 임베딩 가중치 배수. `None`이면 `1.0`.
  - **logits_scaling** (`float`, default `None`) — 로짓 스케일. `None`이면 `1.0`.
  - **attention_multiplier** (`float`, default `None`) — 어텐션 가중치 배수. `None`이면 `self.head_dim ** -0.5`.
  - **residual_multiplier** (`float`, default `None`) — 잔차 연결 스케일. `None`이면 `1.0`.
- Peri-LN
  - **use_post_norm** (`bool`, default `False`) — Peri-Layer Normalization 적용 여부. `True`로 활성화.

## 챗 템플릿

[chat_template_hcx.jinja](chat_template_hcx.jinja) (32B 기준).

- thinking 분기: `{%- if enable_thinking is not defined or enable_thinking is true %}`.
- `enable_thinking` **미지정 시 thinking ON**(`<think>` 생성). `hcx_reasoner.py`
  폴백 기본값도 ON으로 맞춰져 있다.
- 명시적으로 끄려면 `chat_template_kwargs={"enable_thinking": false}`를 보낸다.

## Tool calling

모델은 챗 템플릿이 지시한 아래 XML 포맷으로 함수 호출을 출력한다.

```
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>서울</arg_value>
</tool_call>
```

[hcx_tool_parser.py](parser/hcx_tool_parser.py)가 이 포맷을 OpenAI 호환
`tool_calls`로 변환한다. `<arg_value>`는 문자열이면 원문, 그 외(숫자/불리언/객체)는
JSON으로 들어오므로 `json.loads` 시도 후 실패하면 문자열로 복원한다. 문자열 인자
하나만 담는 `<arguments>{json}</arguments>` 형태도 지원한다.

## 설치

```bash
pip install .   # HyperCLOVAXForCausalLM 및 hcx 파서를 vLLM에 등록
```

## 서버 기동

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model <hcx-model-path> \
    --reasoning-parser hcx \
    --enable-auto-tool-choice --tool-call-parser hcx \
    --chat-template ./chat_template_hcx.jinja
```

`--reasoning-parser hcx`를 함께 켜면 `<think>` 추론부는 reasoning 파서가
처리하고, 추론이 끝난 뒤에야 tool 파서가 호출된다(중복 처리 없음).

## Kubernetes 기동 (git clone 방식)

repo를 클론해 `pip install -e .`로 설치하면 챗 템플릿도 클론 디렉터리에 함께
들어오므로, `--chat-template`에 **클론 경로**(`/tmp/plugin/chat_template_hcx.jinja`)를
그대로 지정한다. 별도 ConfigMap/볼륨 마운트가 필요 없다.

> **EOS 토큰(`100273` 추가):** 모델의 `generation_config.json`을 직접 패치해야 한다
> (`--override-generation-config` 플래그는 적용되지 않음). 모델 마운트가 RW일 때
> vLLM 기동 **전에** 아래 멱등 패치를 실행한다. RO 마운트면 모델 빌드 단계에서 미리
> 반영해야 한다.

```yaml
args:
  - |
    apt-get update && apt-get install -y git

    cd /tmp
    rm -rf plugin
    git clone https://github.com/madcoww/hcx-vllm-plugin-custom.git plugin
    cd plugin
    pip install -e .

    python3 -m vllm.entrypoints.openai.api_server \
      --model /mnt/models/HyperCLOVAX-SEED-Think-32B \
      --served-model-name HyperCLOVAX-SEED-Think-32B \
      --host 0.0.0.0 \
      --port 8000 \
      --gpu-memory-utilization 0.9 \
      --tensor-parallel-size 2 \
      --max-model-len 32768 \
      --max-num-seqs 8 \
      --reasoning-parser hcx \
      --enable-auto-tool-choice \
      --tool-call-parser hcx \
      --disable-custom-all-reduce \
      --chat-template /tmp/plugin/chat_template_hcx.jinja
```

## Docker 배포 ([Docs](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html))

`COPY .`로 챗 템플릿이 이미지 루트(`/workspace/HyperCLOVAX/`)에 함께 들어간다.

```bash
docker build --tag vllm/vllm-openai-hyperclovax .
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai-hyperclovax \
    --model <hcx-model-path> \
    --reasoning-parser hcx \
    --enable-auto-tool-choice --tool-call-parser hcx \
    --chat-template /workspace/HyperCLOVAX/chat_template_hcx.jinja
```

> Docker도 마찬가지로 모델의 `generation_config.json`에 EOS `100273`을 직접 반영해야
> 한다(위 멱등 패치 참고).

## License

```
HyperCLOVAX vLLM Plugin
Copyright (c) 2025-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
