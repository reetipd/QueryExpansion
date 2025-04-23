import os
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.output_parsers import PydanticToolsParser
from langchain.schema import BaseMessage

# Define the data model for paraphrased queries
class ParaphrasedQuery(BaseModel):
    paraphrased_query: str = Field(
        ...,
        description="A unique paraphrasing of the given question."
    )

class QueryExpander:
    """
    A class that uses language models to expand user queries into multiple variations
    while preserving the original intent.
    """
    def __init__(self):
        # Define the system prompt that instructs the LLM how to expand queries
        self.system_prompt = """You are an expert at expanding user questions into multiple variations. \
                Perform query expansion. If there are multiple common ways of phrasing a user question \
                or common synonyms for key words in the question, make sure to return multiple versions \
                of the query with the different phrasings.

                If there are acronyms or words you are not familiar with, do not try to rephrase them.

                Return at least 3 versions of the question that maintain the original intent."""
        
        # Create a prompt template for the conversation
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}")
        ])

        # Initialize the language model using Hugging Face's endpoints
        self.llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",  # Or other free models like mistralai/Mistral-7B-Instruct-v0.1
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
            task="text-generation",
            temperature=0  # Using temperature=0 for more deterministic results
        )

        # Set up the Pydantic tool parser for structured output
        self.query_analyzer = self.llm | PydanticToolsParser(tools=[ParaphrasedQuery])

    def expand_query(self, question):
        """
        Expands a user question into multiple variations using the language model.
        
        Args:
            question (str): The original user question to expand
            
        Returns:
            list: A list of variations of the original question
        """
        try:
            # Format the message for the LLM
            formatted_message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "human", "content": question}
            ]
            
            # Get the response from the LLM
            result = self.llm.invoke(formatted_message)

            # Extract the variations from the result
            variations = [result['choices'][0]['text']]

            return variations
        except Exception as e:
            print(f"Error expanding query: {e}")
            return []
        
def main():
    """
    Main function to run the query expansion tool interactively.
    Handles user input and displays the expanded query variations.
    """
    try:
        # Initialize the query expander
        expander = QueryExpander()

        # Print welcome message and instructions
        print("Welcome to LangChain Query Expander!")
        print("Enter a question to see different variations (or 'quit' to exit)")
        print("\nExample questions:")
        print("1. 'What is the weather like today?'")
        print("2. 'Can you recommend me a good book to read?'")
        print("3. 'How do I make a cup of coffee?'")
        
        # Main interaction loop
        while True:
            # Get user input
            question = input("\nEnter your question: ").strip()

            # Check if user wants to exit
            if question.lower() == 'quit':
                print("Thank you!")
                break

            # Generate variations of the question
            print("\nGenerating variations of the questions...")
            variations = expander.expand_query(question)

            # Display the generated variations
            print("\nGenerated variations:")
            for i, variation in enumerate(variations):
                print(f"{i+1}. {variation}")

            # Show the total number of variations
            print(f"Total number of variations created: {len(variations)}")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your API key and try again.")

# Entry point of the script
if __name__ == "__main__":
    main()