# imports
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

import tiktoken

# set-up
load_dotenv()

# Functions


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result


llm = OpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)

conversation = ConversationChain(
    llm=llm)

print(conversation.prompt.template)

# Memory Types

# 1. ConversationBufferMemory

conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm)
)

conversation_buf('Good Morning AI!')

count_tokens(
    conversation_buf,
    query='My interest here is to explore the potential of integrating LLMs with external knowledge')

count_tokens(
    conversation_buf,
    'I just want to analyze the different possibilities. What can you think of using LangChain?'
)

print(conversation_buf.memory.buffer)

# 2. ConversationSummaryMemory

conversation_sum = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm)
)

print(conversation_sum.prompt.template)

count_tokens(
    conversation_sum,
    query='Good morning AI!'
)

count_tokens(
    conversation_sum,
    query='My interest here is to explore the potential of integrating Large Language MOdels with external knowledge using LangChain'
)

count_tokens(
    conversation_sum,
    query='I just want to analyze the different possibilities. What can you think of?'
)

count_tokens(
    conversation_sum,
    query='Which data source types could be used to give context to the model?'
)

count_tokens(
    conversation_sum,
    'What is my aim again?'
)

print(conversation_sum.memory.buffer)

tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

print(
    f'Buffer memory conversation length: {len(tokenizer.encode(conversation_buf.memory.buffer))}\n'
    f'Summary memory conversation length: {len(tokenizer.encode(conversation_sum.memory.buffer))}'
)

# 3. ConversationBufferWindowMemory

conversation_bufw = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=1)
)

count_tokens(
    conversation_bufw,
    'Good morning AI!'
)

count_tokens(
    conversation_bufw,
    query='My interest here is to explore the potential of integrating LLMs with external knowledge'
)

count_tokens(
    conversation_bufw,
    "I just want to analyze the different possibilities. What can you think of?"
)

count_tokens(
    conversation_bufw,
    "Which data source types could be used to give context to the model?"
)

count_tokens(
    conversation_bufw,
    "What is my aim again?"
)

buffw_history = conversation_bufw.memory.load_memory_variables(
    inputs=[]
)['history']


print(buffw_history)
