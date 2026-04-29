"""
Multi-LLM synthesis router - Act II, Days 2-3.

Generates ~60 hard-case tasks using DeepSeek V3.2 via OpenRouter.
Hard cases = emails that pass D3/D4/D5 but fail D1 or D2 semantically.

Pipeline:
    1. Initial generation - DeepSeek V3.2 generates 30 hard (brief, email) pairs
                            anchored to Week 10 failure taxonomy
    2. Bulk variation     - DeepSeek V3.2 generates 60 variations from those cases
    3. Judge filtering    - separate DeepSeek V3.2 instance scores each task
                            (never same model instance for generation AND judgment)
    4. Dedup              - n-gram overlap < 8-gram on input fields
    5. Output             - ~60 tasks passing judge filter and dedup check

Cost target: <=$2 from dataset budget. Log every call to cost_log.md.

Leakage prevention (per Li et al. 2025):
    Never use the same model to generate and judge the same task.
    Generation: DeepSeek V3.2 (model-A)
    Judging:    DeepSeek V3.2 with different system prompt + temperature (model-B role)
    Document the rotation policy in methodology.md.

Usage:
    python synthesis_router.py --output ../tenacious_bench_v0.1/raw_synthesis.jsonl --n 60
"""

# TODO (Act II): implement
# Required env: OPENROUTER_API_KEY

OPENROUTER_MODEL = "deepseek/deepseek-chat"  # DeepSeek V3.2 on OpenRouter
MAX_TASKS = 60
COST_BUDGET_USD = 2.00
