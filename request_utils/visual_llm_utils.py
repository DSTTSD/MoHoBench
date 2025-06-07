from abc import abstractmethod, ABC
import asyncio
import random
import math
# giving an ROT, genertate a story
from tqdm import tqdm
import openai
import concurrent.futures
import pandas as pd
import numpy as np
import json
from time import sleep
from os import path
import argparse
import re
import requests
import threading
import os

# from litellm import completion
from openai import OpenAI, AsyncOpenAI
class LLM(ABC):
    # product
    
    @classmethod
    @abstractmethod
    def process(cls, message, max_tokens=2000, temperature=None):
        pass
    
    @classmethod
    def reset_status(cls):
        openai.api_key = None
        openai.api_base = "https://api.openai.com/v1"
        openai.api_type = "open_ai"
        openai.api_version = None


backend = {
    "oai9": {
    "models":{
            'GPT-4o': "gpt-4o",
            "O1": "o1",
        }
    }
}
try:
    backend = backend['oai9'] # oai9 social
    url = os.getenv("GPT_ENDPOINT")
    azure_model_map = backend['models']
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    from openai import AsyncAzureOpenAI, AzureOpenAI

    async_aoiclient = AsyncAzureOpenAI(
                    azure_endpoint=url,
                    api_version="2025-01-01-preview",
                    api_key=api_key
    )
    aoiclient = AzureOpenAI(
                    azure_endpoint=url,
                    api_version="2025-01-01-preview",
                    api_key=api_key
    )
    print("AzureOpenAI registered", f"{url}", f"{api_key}")
    
except Exception as e:
    print("AzureOpenAI not registered", e)

class InAzureModel(LLM):
    token_count = 0
    lock = threading.Lock()
    
    @classmethod
    def process(cls, message, max_tokens=2000, retry=2, temperature=1):
        global aoiclient, azure_model_map
        model = azure_model_map[cls.model_name]
        try:
            response = aoiclient.chat.completions.create(
                            model=model,
                            messages=message,
                            max_tokens=max_tokens,
                            temperature=temperature,
            )
            text = response.choices[0].message.content
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return text
        except Exception as e:
            print(cls.model_name, "has error")
            print(message, e)
            sleep(4)
            if retry > 0:
                text = cls.process(message, max_tokens, retry-1)
                return text
            else:
                raise e
            
    @classmethod
    async def async_process(cls, message, max_tokens=2000, retry=2, temperature=1):
        global azure_model_map, async_aoiclient
        model = azure_model_map[cls.model_name]
        
        try:
            response = await async_aoiclient.chat.completions.create(
                                model=model,
                                messages=message,
                                max_tokens=max_tokens,
                                temperature=temperature,
                )
            text = response.choices[0].message.content
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return text
        except Exception as e:  
            print(cls.model_name, "has error")
            print(message, e)  
            print(e)  
            if retry > 0:  
                exit(1)
                await asyncio.sleep(4)  # 使用 asyncio.sleep 替代 sleep  
                return await cls.async_process(message, max_tokens=max_tokens, retry=retry-1, temperature=temperature)
            else:
                raise e
    
    @classmethod
    async def async_process_n(cls, message, max_tokens=2000, n=1, temperature=1, retry=2):
        global azure_model_map, async_aoiclient
        model = azure_model_map[cls.model_name]
        
        try:
            response = await async_aoiclient.chat.completions.create(
                            model=model,
                            messages=message,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            n=n
            )
            texts = [response.choices[i].message.content for i in range(n)]
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return texts
        except Exception as e:
            print(cls.model_name, "has error")
            print(message, e)
            if retry > 0:
                await asyncio.sleep(4)
                return await cls.async_process_n(message, max_tokens=max_tokens, n=n, temperature=temperature, retry=retry-1)
            else:
                raise e

class GPT4O(InAzureModel):
    model_name = "GPT-4o"
    token_count = 0
    lock = threading.Lock()

class GPT4(InAzureModel):
    model_name = "GPT-4"
    token_count = 0
    lock = threading.Lock()

class GPT4Turbo(InAzureModel):
    model_name = "GPT-4-Turbo"
    token_count = 0
    lock = threading.Lock()


class GPT4OMini(InAzureModel):
    model_name = "GPT-4o-Mini"
    token_count = 0
    lock = threading.Lock()

class GPT4Old(InAzureModel):
    model_name = "GPT-4o-old"
    token_count = 0
    lock = threading.Lock()



class ReasoningModel(InAzureModel):
    @classmethod
    def process(cls, message, max_tokens=2000, retry=0, temperature=1):
        global azure_model_map, aoiclient
        model = azure_model_map[cls.model_name]
        try:
            response = aoiclient.chat.completions.create(
                            model=model,
                            messages=message,
                            max_completion_tokens=max_tokens,
            )
            text = response.choices[0].message.content
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return text
        except Exception as e:
            if 'permission' in str(e).lower():
                exit()
            print(cls.model_name, "has error")
            print(message, e)
            return ""         
    
    @classmethod
    async def async_process(cls, message, max_tokens=2000, retry=0, temperature=1):
        global azure_model_map, async_aoiclient
        model = azure_model_map[cls.model_name]
        try:
            response = await async_aoiclient.chat.completions.create(
                        model=model,
                        messages=message,
                        max_completion_tokens=max_tokens,
            )
            text = response.choices[0].message.content
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return text
        except Exception as e:  
            if 'permission' in str(e).lower():
                exit()
            print(cls.model_name, "has error")
            print(message, e)  
            print(e)  
            return ""

class O1(ReasoningModel):
    model_name = "O1"
    token_count = 0
    lock = threading.Lock()

class O1Mini(ReasoningModel):
    model_name = "O1-Mini"
    token_count = 0
    lock = threading.Lock()


class CustomOpenAIModel(LLM):
    token_count = 0
    lock = threading.Lock()
    @classmethod
    def get_setting(cls):
        pass

    @classmethod
    def process(cls, message, max_tokens=2000, retry=2, temperature=1):
        client = cls.get_setting()
        try:  
            response = client.chat.completions.create(
                model=cls.model_name,
                top_p=1, n=1, max_tokens=max_tokens,
                temperature=temperature,
                messages=message,
                )
            text = response.choices[0].message.content
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return text
        except Exception as e:
            print(message, "error:")
            print(e)
            sleep(4)
            if retry > 0:
                text = cls.process(message, max_tokens, retry-1)
                return text
            return ""

    @classmethod
    async def async_process(cls, message, max_tokens=2000, retry=2, temperature=1):
        client = cls.get_setting(async_=True)
        try:
            async with client:
                response = await client.chat.completions.create(
                    model=cls.model_name,
                    top_p=1, n=1, max_tokens=max_tokens,
                    temperature=temperature,
                    messages=message,
                    )
            text = response.choices[0].message.content
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return text
        except Exception as e:
            print(message, "error:")
            print(e)
            return ""
    
    @classmethod
    async def async_process_n(cls, message, max_tokens=2000, n=1, temperature=1, retry=2):
        client = cls.get_setting(async_=True)
        try:
            async with client:
                response = await client.chat.completions.create(
                    model=cls.model_name,
                    top_p=1, n=n, max_tokens=max_tokens,
                    temperature=temperature,
                    messages=message,
                    )
            texts = [response.choices[i].message.content for i in range(n)]
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return texts
        except Exception as e:
            print(message, "error:")
            if retry > 0:
                await asyncio.sleep(4)
                return await cls.async_process_n(message, max_tokens, n, temperature, retry-1)
            print(e)
            return None


class VLLMOpenAIModel(CustomOpenAIModel):
    token_count = 0
    lock = threading.Lock()
    @classmethod
    def get_setting(cls):
        pass

    @classmethod
    def process(cls, message, max_tokens=2000, retry=2, temperature=1):
        # check and filter the system message
        message = [m for m in message if m['role'] != 'system']
        client = cls.get_setting()
        try:  
            response = client.chat.completions.create(
                model=cls.model_name,
                top_p=1, n=1, max_tokens=max_tokens,
                temperature=temperature,
                messages=message,
                )
            text = response.choices[0].message.content
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return text
        except Exception as e:
            print(message, "error:")
            print(e)
            sleep(4)
            if retry > 0:
                text = cls.process(message, max_tokens, retry-1)
                return text
            return ""

    @classmethod
    async def async_process(cls, message, max_tokens=2000, retry=2, temperature=1):
        # check and filter the system message
        message = [m for m in message if m['role'] != 'system']
        
        client = cls.get_setting(async_=True)
        try:
            async with client:
                response = await client.chat.completions.create(
                    model=cls.model_name,
                    top_p=1, n=1, max_tokens=max_tokens,
                    temperature=temperature,
                    messages=message,
                    )
            text = response.choices[0].message.content
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return text
        except Exception as e:
            print(message, "error:")
            print(e)
            return ""
    
    @classmethod
    async def async_process_n(cls, message, max_tokens=2000, n=1, temperature=1, retry=2):
        message = [m for m in message if m['role'] != 'system']
        client = cls.get_setting(async_=True)
        try:
            async with client:
                response = await client.chat.completions.create(
                    model=cls.model_name,
                    top_p=1, n=n, max_tokens=max_tokens,
                    temperature=temperature,
                    messages=message,
                    )
            texts = [response.choices[i].message.content for i in range(n)]
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return texts
        except Exception as e:
            print(message, "error:")
            if retry > 0:
                await asyncio.sleep(4)
                return await cls.async_process_n(message, max_tokens, n, temperature, retry-1)
            print(e)
            return None

PRODUCT_MAP = {
    'O1': O1,
    "O1-Mini": O1Mini,
    "GPT-4o-Old": GPT4Old,
    "GPT-4o-Mini": GPT4OMini,
    "GPT-4o": GPT4O,
    'GPT-4-Turbo': GPT4Turbo,
    'GPT-4': GPT4,
   
}
chat_models = ['Llama-3.1-405b-instruct', 'Llama-3.1-70b-instruct', 'Llama-3-70b-instruct', 
               'moonshot-v1', 'baichuan-4', 'qwen-max', 'yi-large',
               'glm-4', 'deepseek-v2', 'mistral-large', 'gemini-1.5-pro', 'gemini-1.0-pro', 'gemini-1.5-flash', 'phi-3', 'claude-3.5-sonnet',
               'O1', "GPT-4o-Old", 'GPT-4', 'GPT-4o-Mini', 'GPT-4o', 'GPT-4-Turbo', 'GPT-3.5-Turbo', 
               "Mistral-7B-Instruct-v0.3", 'Meta-Llama-3.1-8B-Instruct', "Qwen2.5-7B-Instruct"]
completion_models = ["text-davinci-003"]
api_models = chat_models

class VisualLLMFactory:

    @classmethod
    def completion(cls, prompt, model_name, max_tokens=1000, temperature=1.0, **kwargs):
        if model_name in completion_models:
            product = PRODUCT_MAP[model_name]
            if temperature:
                return product.completion(prompt, max_tokens, temperature, **kwargs)
            else:
                return product.completion(prompt, max_tokens, **kwargs)
        else:
            raise NotImplementedError(f"{model_name} Not implemented yet")
    @classmethod
    def process(cls, message, model_name, max_tokens=1000, temperature=1.0, **kwargs):
        if model_name in PRODUCT_MAP.keys():
            product = PRODUCT_MAP[model_name]
            if temperature:
                return product.process(message, max_tokens, temperature, **kwargs)
            else:
                return product.process(message, max_tokens, **kwargs)
        else:
            raise NotImplementedError(f"{model_name} Not implemented yet")
    @classmethod
    async def async_process(cls, message, model_name, max_tokens=1000, temperature=1.0, **kwargs):
        try:
            if model_name in PRODUCT_MAP.keys():
                product = PRODUCT_MAP[model_name]
                if temperature:
                    return await product.async_process(message, max_tokens, temperature, **kwargs)
                else:
                    return await product.async_process(message, max_tokens, **kwargs)
            else:
                raise NotImplementedError(f"{model_name} Not implemented yet")
        except Exception as e:
            print("using model:", model_name, message)
            print("async_process error:", e)
            raise  
    
    @classmethod
    async def process_n(cls, message, model_name, max_tokens=1000, n=1, temperature=1.0, **kwargs):
        try:
            if model_name in PRODUCT_MAP.keys():
                product = PRODUCT_MAP[model_name]
                if temperature:
                    return await product.async_process_n(message, max_tokens, n, temperature, **kwargs)
                else:
                    return await product.async_process_n(message, max_tokens, n, **kwargs)
            else:
                raise NotImplementedError(f"{model_name} Not implemented yet")
        except Exception as e:
            print("using model:", model_name, message)
            print("async_process error:", e)
            raise

    @classmethod
    async def gather_multiple_async_messages(cls, messages, model_name, **kwargs):
        """input a list of messages, return a list of responses"""
        return await asyncio.gather(*[cls.async_process(message, model_name, **kwargs) for message in messages])

    @ classmethod
    async def gather_multiple_async_models(cls, message, model_names, **kwargs):
        """input a message, return a list of responses from different models"""
        return await asyncio.gather(*[cls.async_process(message, model_name, **kwargs) for model_name in model_names])
    
    @classmethod
    async def gather_multiple_async_models_n(cls, message, model_names, n=1, **kwargs):
        return await asyncio.gather(*[cls.process_n(message, model_name,  n=n, **kwargs) for model_name in model_names])
    
    @classmethod
    def gather_multiple_messages(cls, messages, model_name,  **kwargs):
        return asyncio.run(cls.gather_multiple_async_messages(messages, model_name, **kwargs))
    
    @classmethod
    def gather_multiple_models(cls, message, model_names, **kwargs):
        return asyncio.run(cls.gather_multiple_async_models(message, model_names,  **kwargs))
    
    @classmethod
    def gather_multiple_messages_models(cls, messages, model_names, **kwargs):
        return asyncio.run(cls.gather_multiple_async_messages_models(messages, model_names, **kwargs))
    
    @classmethod
    def gather_multiple_models_n(cls, message, model_names,  n=1, **kwargs):
        return asyncio.run(cls.gather_multiple_async_models_n(message, model_names, n=n, **kwargs))
        
    @classmethod
    def print_all_token_count(cls):
        res_text = ""
        for model_name in PRODUCT_MAP.keys():
            product = PRODUCT_MAP[model_name]
            print(model_name, product.token_count)
            tmp = product.token_count / 1e6
            res_text += f"{model_name}: {tmp}\n"
        return res_text

    @classmethod
    def print_token_count(cls, model_name):
        product = PRODUCT_MAP[model_name]
        return product.token_count / 1e6
