MISTRAL_CHAT_TEMPLATE = """{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}

{{- bos_token }}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif %}
    {%- if message['role'] == 'user' %}
        {%- if loop.first and system_message is defined %}
            {{- ' [INST] ' + system_message + '\n\n' + message['content'] + ' [/INST]' }}
        {%- else %}
            {{- ' [INST] ' + message['content'] + ' [/INST]' }}
        {%- endif %}
    {%- elif message['role'] == 'assistant' %}
        {{- ' ' + message['content'] + eos_token}}
    {%- else %}
        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}
    {%- endif %}
{%- endfor %}"""


PROMPT_ONE_CONTEXT = """Answer my questions based on the given context.
---
## Context
{context}
---
## Your Task
{answer_instruction}
Question: {question}
Your Answer:"""

PROMPT_MISTRAL = """Answer my questions based on the given context.
---
## Context
{context}
---
## Additional Context
{additional_context}
---
## Your Task
{answer_instruction}
Question: {question}
Your Answer:"""

PROMPT_QUESTION_ONLY = """You are an assistant for giving short answers.
{answer_instruction}
---
Question: {question}
"""

PROMPT_QUESTION_AND_GOLDEN_CONTEXT = """You are an assistant for giving short answers based on the given context.
## Context
---
{context}
---
{answer_instruction}
Question: {question}
"""



PROMPT_LLAMA3 = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Answer my questions based on the given context.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
## Context
---
{context}
---
{additional_context}
{answer_instruction}
Question: {question} <|eot_id|>
<|start_header_id|>Assistant:<|end_header_id|>"""

PROMPT_LLAMA3_QUESTION_AND_GOLDEN_CONTEXT = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an assistant for giving short answers based on the given context.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
## Context
---
{context}
---
{answer_instruction}
Question: {question} <|eot_id|>
<|start_header_id|>Assistant:<|end_header_id|>"""


PROMPT_LLAMA3_QUESTION_ONLY = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an assistant for giving short answers.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{answer_instruction}
Question: {question}<|eot_id|>
<|start_header_id|>Assistant:<|end_header_id|>"""

ANSWER_INSTRUCTION = "Answer the following question in a succinct manner. Use a single phrase or a short sentence if possible."
ANSWER_INSTRUCTION_MULTI_CHOICE = "Answer the following question with the option's letter from the given choices directly, such as A, B, C, or D. Do not add any other text."
# ANSWER_INSTRUCTION = "Answer the following question briefly."


PROMPT_HF_MODELS = """
You are an assistant for giving short answers based on the given context.
## Context
---
{context}
---
{answer_instruction}
Question: {question}
Your Answer:
"""

PROMPT_HF_MODELS_QUESTION_ONLY = """
You are an assistant for giving short answers.
{answer_instruction}
Question: {question}
Your Answer:
"""

prompt_reformat_data = """We are training our chatbot. You are tasked with reformatting the answers in a dataset into instruction-following formats based on the question, original answers, and the context. 

## Golden Context
{context}

## Question
{question}

## Original Answer
{answer}

## Your task
Reformat the `Original Answer` into coherent sentences based on the question and the golden context. Return your answer simply as a string:
"""

prompt_reformat_data_no_context = """We are training our chatbot. You are tasked with reformatting the answers in a dataset into instruction-following formats based on the question and the original answers. 

## Question
{question}

## Original Answer
{answer}

## Your task
Reformat the `Original Answer` into coherent sentences based on the question. Return your answer simply as a string:
"""



