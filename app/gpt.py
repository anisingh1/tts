
import os  
from openai import AzureOpenAI  


class GPT:
    def __init__(self):
        endpoint = os.getenv("ENDPOINT_URL", "https://gpt4-g11n.openai.azure.com/")  
        self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt4o")  
        subscription_key = os.getenv("AZURE_OPENAI_KEY")  

        # Initialize Azure OpenAI Service client with key-based authentication    
        self.client = AzureOpenAI(  
            azure_endpoint=endpoint,  
            api_key=subscription_key,  
            api_version="2024-05-01-preview",
        )
    
        #Prepare the chat prompt 
        self.chat_prompt = [
            {
                "role": "system",
                "content": "find the syllable count in the user text"
            }
        ] 
    
    
    def generate(self, text: str):
        messages = self.chat_prompt
        messages.append({"role": "user", "content": text})
        completion = self.client.chat.completions.create(  
            model=self.deployment,
            messages=messages,
            max_tokens=800,  
            temperature=0.7,  
            top_p=0.95,  
            frequency_penalty=0,  
            presence_penalty=0,
            stop=None,  
            stream=False
        )
        return completion.to_json()
