# imports
import inspect
import re

from dotenv import load_dotenv
from getpass import getpass
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import (
    LLMChain,
    LLMMathChain,
    TransformChain,
    SequentialChain
)
from langchain.callbacks import get_openai_callback
from langchain.chains import load_chain

# set-up
load_dotenv()

llm = OpenAI(temperature=0, model_name='text-davinci-003')

# helper functions


def count_tokens(chain, query: str) -> str:
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens.')

    return result


# Utility chains
llm_math = LLMMathChain(llm=llm, verbose=True)

count_tokens(llm_math, 'What is 13 raised to the power of 10?')

print(llm_math.prompt.template)

# we set the prompt to only have the question we ask
prompt = PromptTemplate(
    input_variables=['question'],
    template='{question}'
)
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

count_tokens(llm_chain, 'What is the square root of 4?')

print(inspect.getsource(llm_math._call))

# Generic Chains


def transform_func(inputs: dict) -> dict:
    text = inputs['text']

    text = re.sub(r'(\r\n|\r|\n){2,}', r'\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return {'output_text': text}


clean_extra_spaces_chain = TransformChain(
    input_variables=['text'],
    output_variables=['output_text'],
    transform=transform_func
)

clean_extra_spaces_chain.run(
    'A random text  with   some irregular spacing.\n\n\n     Another one   here as well.')


template = '''
Paraphase this text:

{output_text}

In the style of a {style}.

Paraphase:
'''

prompt = PromptTemplate(
    input_variables=['style', 'output_text'],
    template=template
)


style_paraphrase_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_key='final_output'
)

sequential_chain = SequentialChain(
    chains=[
        clean_extra_spaces_chain,
        style_paraphrase_chain
    ],
    input_variables=['text', 'style'],
    output_variables=['final_output']
)

input_text = '''
Chains allow us to combine multiple 


components together to create a single, coherent application. 

For example, we can create a chain that takes user input,       format it with a PromptTemplate, 

and then passes the formatted response to an LLM. We can build more complex chains by combining     multiple chains together, or by 


combining chains with other components.
'''

count_tokens(
    sequential_chain,
    {'text': input_text,
     'style': 'a medieval icelandic poetry'}
)

# langchain-hub
llm_math_chain = load_chain('lc://chains/llm-math/chain.json')

llm_math_chain.verbose

llm_math_chain.verbose = False

