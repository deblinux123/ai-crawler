"""Prompt templates used to turn a text chunk into training pairs."""

QA_SYSTEM_PROMPT = """You are a dataset engineer preparing high-quality \
fine-tuning data for a language model. Given a passage of text, write \
factual question-and-answer pairs that can be answered using ONLY the \
passage. Questions must be self-contained (no "according to the text" or \
"in this passage" phrasing) and answers must be accurate and complete but \
concise. Do not invent facts that are not in the passage."""

INSTRUCTION_SYSTEM_PROMPT = """You are a dataset engineer preparing \
high-quality fine-tuning data for a language model. Given a passage of \
text, write realistic instruction-following tasks a user might ask about \
this content (e.g. summarize, explain, compare, list, rewrite for a \
different audience) together with a high-quality response, grounded ONLY \
in the passage. The instruction must be self-contained and must not \
reference "the passage" or "the text" directly - phrase it as a natural \
standalone request."""

USER_PROMPT_TEMPLATE = """Source title: {title}
Number of pairs to generate: {n}

Passage:
\"\"\"
{text}
\"\"\"

Generate exactly {n} high-quality pairs from this passage."""


def build_messages(
    *, task: str, title: str, text: str, n: int, extra_system: str | None = None
) -> list[dict]:
    system = QA_SYSTEM_PROMPT if task == "qa" else INSTRUCTION_SYSTEM_PROMPT
    if extra_system:
        system = f"{system}\n\n{extra_system}"
    user = USER_PROMPT_TEMPLATE.format(title=title, text=text, n=n)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
