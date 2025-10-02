import json
import textwrap
import google.generativeai as genai
import pandas as pd

from abc import ABC, abstractmethod
from openai import OpenAI
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import JsonOutputParser
from google.generativeai.types.generation_types import GenerationConfig



def parse_json_markdown(json_string: str) -> dict:
    """
    Parse JSON string from markdown format
    Args:
        json_string (str): JSON string in markdown format
    
    Returns:
        dict: Parsed JSON dictionary
    """
    try:
        # Try to find JSON string within first and last triple backticks
        if json_string[3:13].lower() == "typescript":
            json_string = json_string.replace(json_string[3:13], "",1)
        
        if 'JSON_OUTPUT_ACCORDING_TO_RESUME_DATA_SCHEMA' in json_string:
            json_string = json_string.replace("JSON_OUTPUT_ACCORDING_TO_RESUME_DATA_SCHEMA", "",1)
        
        if json_string[3:7].lower() == "json":
            json_string = json_string.replace(json_string[3:7], "",1)
    
        parser = JsonOutputParser()
        parsed = parser.parse(json_string)

        return parsed
    except Exception as e:
        print(e)
        return None
    

class LLMProvider(ABC):
    """Abstract class for Language Model Provider
    Requires implementation of get_response and get_embedding methods
    """
    def __init__(self, api_key, model, system_prompt, max_output_tokens=None, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    @abstractmethod
    def get_response(self, prompt, need_json_output=False):
        pass

    @abstractmethod
    def get_embedding(self, text, model=None, task_type="retrieval_document"):
        pass


class ChatGPT(LLMProvider):
    """
    LLMProvider implementation using OpenAI's ChatCompletion API.
    Automatically retries without custom temperature if unsupported by the model.
    """
    def __init__(self, api_key, model, system_prompt, max_output_tokens=None, temperature=0.7):
        super().__init__(api_key, model, system_prompt, max_output_tokens, temperature)
        self.embedding_model = "text-embedding-ada-002"
        if self.system_prompt.strip():
            self.system_prompt = {"role": "system", "content": self.system_prompt}
        self.client = OpenAI(api_key=self.api_key)
    
    def get_response(self, prompt, need_json_output=False):
        """
        Get a chat response from the OpenAI model.
        Retries without temperature if the model rejects custom temperature values.

        Args:
            prompt (str): The prompt string to send as user content.
            need_json_output (bool): Whether to parse the response as JSON.

        Returns:
            str or dict: The model's response content, parsed JSON if requested.
        """
        user_prompt = {"role": "user", "content": prompt}

        # Build request parameters without sending null or unsupported values
        params = {
            "model": self.model,
            "messages": [self.system_prompt, user_prompt],
        }
        if self.max_output_tokens is not None:
            params["max_tokens"] = self.max_output_tokens
        if need_json_output:
            params["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            params["temperature"] = self.temperature

        try:
            completion = self.client.chat.completions.create(**params)
        except Exception as e:
            err = str(e)
            # Retry without temperature if unsupported by the model
            if "Unsupported value: 'temperature'" in err:
                params.pop("temperature", None)
                completion = self.client.chat.completions.create(**params)
            else:
                raise Exception(f"Error in ChatGPT API, {e}")

        response = completion.choices[0].message
        content = response.content.strip()

        if need_json_output:
            return parse_json_markdown(content)
        return content
        
    def get_embedding(self, text, model=None, task_type="retrieval_document"):
        """
        Get embeddings for the given text using OpenAI embeddings API.

        Args:
            text (str): The text to embed.
            model (str, optional): The embedding model to use. Defaults to None.
            task_type (str): The task type for the embedding call.

        Returns:
            list[float]: The embedding vector for the input text.
        """
        try:
            if model is None:
                model = self.embedding_model
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        except Exception as e:
            raise Exception(f"Error in ChatGPT API, {e}")

class Gemini(LLMProvider):
    def __init__(self, api_key, model, system_prompt, max_output_tokens=None, temperature=0.7):
        super().__init__(api_key, model, system_prompt, max_output_tokens, temperature)
        self.embedding_model = "models/text-embedding-004"
        genai.configure(api_key=self.api_key)
    
    def get_response(self, prompt, need_json_output=False):
        try:
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=self.system_prompt
                )
            
            content = model.generate_content(
                contents=prompt,
                generation_config=GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens = self.max_output_tokens,
                    response_mime_type = "application/json" if need_json_output else None
                    )
                )

            if need_json_output:
                result = parse_json_markdown(content.text)
            else:
                result = content.text
            
            return result
        
        except Exception as e:
            print(e)
            return None
    
    def get_embedding(self, content, model=None, task_type="retrieval_document"):
        try:
            if model is None:
                model = self.embedding_model
            def embed_fn(data):
                result = genai.embed_content(
                    model=model,
                    content=data,
                    task_type=task_type,
                    title="Embedding of json text" if task_type in ["retrieval_document", "document"] else None)
                
                return result['embedding']
            
            df = pd.DataFrame(content)
            df.columns = ['chunk']
            df['embedding'] = df.apply(lambda row: embed_fn(row['chunk']), axis=1)
            
            return df
        
        except Exception as e:
            print(e)

class OllamaModel(LLMProvider):
    def __init__(self, model, system_prompt, max_output_tokens=None, temperature=0.7):
        super().__init__(api_key=None, model=model, system_prompt=system_prompt, max_output_tokens=max_output_tokens, temperature=temperature)
        self.embedding_model = "bge-m3"
    
    def get_response(self, prompt, need_json_output=False):
        try:
            llm = Ollama(
                model=self.model, 
                system=self.system_prompt,
                temperature=self.temperature, 
                top_p=0.999, 
                top_k=250,
                num_predict=self.max_output_tokens,
                # format='json' if need_json_output else None,
                )
            content = llm.invoke(prompt)

            if need_json_output:
                result = parse_json_markdown(content)
            else:
                result = content
            
            if result is None:
                return None
            else:
                return result
        except Exception as e:
            print(e)
            return None