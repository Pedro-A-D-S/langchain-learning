from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv

load_dotenv()

davinci = OpenAI(model_name='text-davinci-003')

template = '''
Question: {question}
Answer:
'''

prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

question = 'When was publish the book called A Game of Thrones by George R. R. Martin?'

print(llm_chain.run(question))

qs = [
    {'question': 'When was publish the book called A Game of Thrones by George R. R. Martin?'},
    {'question': 'How long did it take him to write the first book?'},
    {'question': 'In which order should I read his A Song of Ice and Fire books?'}
]

print(llm_chain.generate(qs))

qs_str = (
    "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?" +
    "How many eyes does a blade of grass have?"
)

print(llm_chain.run(qs_str))