from OPENAI_API_KEY import openapi_key
import os
os.environ['OPENAI_API_KEY'] = openapi_key

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(temperature=0.6)

# Prompt 1
name_prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest a good name."
)

# Prompt 2
menu_prompt = PromptTemplate(
    input_variables=["restaurant_name"],
    template="Suggest some menu items for {restaurant_name}. Return as comma-separated list."
)

# Generate restaurant name
restaurant_chain = name_prompt | llm

# Sequential chain
chain = (
    RunnablePassthrough()
    .assign(
        restaurant_name=restaurant_chain
    )
    .assign(
        menu_items=menu_prompt | llm
    )
)

response = chain.invoke({"cuisine": "Mexican"})

print("Restaurant Name:")
print(response["restaurant_name"].content)

print("\nMenu Items:")
print(response["menu_items"].content)