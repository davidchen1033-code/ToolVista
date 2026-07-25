#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp}"
TOOLVQA_ROOT="${TOOLVQA_ROOT:-${ROOT}/ToolVQA-release}"
TOOLVQA_ENV="${TOOLVQA_ENV:-${ROOT}/venvs/toolvqa}"
BASE_PYTHON="${BASE_PYTHON:-/root/miniconda3/bin/python}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}"
HF_HOME="${HF_HOME:-${ROOT}/cache/huggingface}"
TOOLVQA_HF_ENDPOINT="${TOOLVQA_HF_ENDPOINT:-https://hf-mirror.com}"

test -x "${BASE_PYTHON}"
test -f "${TOOLVQA_ROOT}/opencompass/run.py"
test -d "${TOOLVQA_ROOT}/agentlego/agentlego"

if [[ ! -x "${TOOLVQA_ENV}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${TOOLVQA_ENV}"
fi

TOOLVQA_PYTHON="${TOOLVQA_ENV}/bin/python"
"${TOOLVQA_PYTHON}" -m pip install --index-url "${PIP_INDEX_URL}" --upgrade \
  pip wheel 'setuptools>=70'

"${TOOLVQA_PYTHON}" -m pip install --index-url "${PIP_INDEX_URL}" \
  --upgrade-strategy only-if-needed \
  'numpy==1.26.4' \
  'pandas==2.3.3' \
  'transformers==4.57.6' \
  'huggingface-hub==0.36.2' \
  'datasets==4.8.4' \
  'sentence-transformers==5.6.0' \
  'scipy==1.13.1' \
  'scikit-learn==1.5.2' \
  'mmengine-lite==0.10.7' \
  'phx-class-registry==4.1.0' \
  'griffe==0.48.0' \
  'duckduckgo-search==5.3.1b1' \
  'cachetools==5.4.0' \
  'openapi-pydantic==0.4.1' \
  'openai==2.45.0' \
  'tiktoken==0.13.0' \
  'opencv-python-headless==4.10.0.84' \
  'tabulate==0.9.0' \
  'prettytable==3.10.2' \
  'rank-bm25==0.2.2' \
  'jsonlines==4.0.0' \
  'jieba==0.42.1' \
  'evaluate==0.4.6' \
  'addict==2.4.0' \
  'func-timeout==4.3.5' \
  'timeout-decorator==0.5.0' \
  'nltk==3.9.1' \
  'rouge==1.0.1' \
  'rouge-chinese==1.0.3' \
  'rouge-score==0.1.2' \
  'sacrebleu==2.5.1' \
  'termcolor==3.3.0' \
  'importlib-metadata==8.7.1'

# Lagent declares Jupyter and Streamlit as hard dependencies, but the official
# ToolVQA OpenCompass step evaluator does not import those UI packages.
"${TOOLVQA_PYTHON}" -m pip install --index-url "${PIP_INDEX_URL}" \
  --no-deps 'lagent==0.2.3'

TOOLVQA_SITE="$("${TOOLVQA_PYTHON}" -c \
  'import site; print(site.getsitepackages()[0])')"
printf '%s\n' \
  "import sys; sys.path[:0] = ['${TOOLVQA_ROOT}/opencompass', '${TOOLVQA_ROOT}/agentlego']" \
  > "${TOOLVQA_SITE}/toolvqa_local_fork.pth"

env \
  HF_ENDPOINT="${TOOLVQA_HF_ENDPOINT}" \
  HF_HOME="${HF_HOME}" \
  HF_HUB_DISABLE_XET=1 \
  "${TOOLVQA_PYTHON}" -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

env \
  HF_HOME="${HF_HOME}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  "${TOOLVQA_PYTHON}" -c '
from lagent.agents import ReAct
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn
from opencompass.datasets.gta_bench import GTABenchDataset, GTABenchEvaluator
from opencompass.models import OpenAIVisual
from opencompass.models.lagent import LagentAgentVisual
from opencompass.openicl.icl_inferencer import AgentInferencer
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-mpnet-base-v2", local_files_only=True)
print("TOOLVQA_RUNTIME_READY")
'

echo "ToolVQA evaluator environment: ${TOOLVQA_ENV}"
echo "ToolVQA Python: ${TOOLVQA_PYTHON}"
echo "Hugging Face cache: ${HF_HOME}"
