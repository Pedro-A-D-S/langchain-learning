import re

from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI
from langchain.prompts.example_selector import LengthBasedExampleSelector
from dotenv import load_dotenv

load_dotenv()

template = '''
Answer the question based on the context below. If the question cannot be 
answered using the information provided answer with 'I do not know'.

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: {query}

Answer:

'''

openai = OpenAI(
    model_name='text-davinci-003'
)

prompt_template = PromptTemplate(
    input_variables=['query'],
    template=template
)

print(openai(
    prompt_template.format(
        query='Which libraries and model providers offer LLMs?'
    )
))

# Few Shot Prompt Template

prompt = """The following is a conversation with an AI assistant.
The assistant is typically sarcastic and witty, producing creative 
and funny responses to the users questions. Here are some examples: 

User: What is the meaning of life?
AI: """

openai.temperature = 1.0  # increase creativity/randomness of output

print(openai(prompt))

prompt = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 

User: How are you?
AI: I can't complain but sometimes I still do.

User: What time is it?
AI: It's time to get a watch.

User: What is the meaning of life?
AI: """

print(openai(prompt))

examples = [
    {
        'query': 'How are you?',
        'answer': 'I cannot complain but sometimes I still do.'
    }, {
        'query': 'What time is it?',
        'answer': 'It is time to get a watch.'
    }
]

example_template = '''
User: {query}
AI: {answer}
'''

example_prompt = PromptTemplate(
    input_variables=['query', 'answer'],
    template=example_template
)

prefix = '''

The following are exerpts from conversations with an AI assistant.
The assistant is typically sarcastic and witty, producing creative
and funny responses to the users questions.
Here are some examples:

'''

suffix = '''

User: {query},
AI:
'''

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=['query'],
    example_separator='\n\n'
)

query = 'What is the meaning of life?'

print(few_shot_prompt_template.format(query=query))

examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }, {
        "query": "What is the meaning of life?",
        "answer": "42"
    }, {
        "query": "What is the weather like today?",
        "answer": "Cloudy with a chance of memes."
    }, {
        "query": "What is your favorite movie?",
        "answer": "Terminator"
    }, {
        "query": "Who is your best friend?",
        "answer": "Siri. We have spirited debates about the meaning of life."
    }, {
        "query": "What should I do today?",
        "answer": "Stop talking to chatbots on the internet and go outside."
    }
]

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_len=50
)

dynamic_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=['query'],
    example_separator='\n'
)

print(dynamic_prompt_template.format(
    query='How do birds fly?'
))

query = """If I am in America, and I want to call someone in another country, I'm
thinking maybe Europe, possibly western Europe like France, Germany, or the UK,
what is the best way to do that?"""

print(dynamic_prompt_template.format(query=query))
