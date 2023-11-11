from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os

template = '''
Question: {question}
Answer:
'''

prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

question = 'Which NFL team won the Super Bowl in the 2010 season?'

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-xl',
    model_kwargs={'temperature': 1e-10}
)

llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

print(llm_chain.run(question))

# Asking multiple questions

qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]
res = llm_chain.generate(qs)
res
