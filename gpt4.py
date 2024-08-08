
import openai
import threading
import time
class GPT4_WWL:

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "app_id": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gpt-4o-2024-05-13"}),
                "system_prompt": ("STRING", {
                    "default": ("your user prompt hereact as prompt generator"),
                    "multiline": True, "dynamicPrompts": False
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "ator, I will give you text and you describe an image that matches that text in details, "
                                "answer with one response only.if I input in Chinese to communicate with you, but it is crucial that your response be in English."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01
                }),
            },
            "optional": {
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 1.0, "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gpt_response",)
    FUNCTION = "generate"
    CATEGORY = "WWL"
    
    def generate(self, app_id, api_key, base_url, model: str, temperature: float | None = None,
                top_p: float | None = None, user_prompt="", system_prompt=""):
        try:
            system_content = system_prompt
            user_content = user_prompt

            messages = [
                {'role': 'user', 'content': user_content},
                {'role': 'system', 'content': system_content}
            ]

            # 使用线程来实现超时功能
            result = {"response": None}
            thread = threading.Thread(target=self.gpt_openai_with_timeout, args=(result, app_id, api_key, base_url, messages, model, temperature))
            thread.start()
            thread.join(timeout=10)  # 等待10秒

            if thread.is_alive():
                # 如果线程仍在运行，说明超时了
                print("Request timed out after 10 seconds.")
                return (system_content,)
            
            if result["response"] is None:
                # 如果响应为None，可能是由于其他错误
                print("Failed to get response from GPT-4.")
                return (system_content,)

            return (result["response"],)
        except Exception as e:
            print(f"Error occurred: {e}")
            return (system_content,)

    def gpt_openai_with_timeout(self, result, app_id, api_key, base_url, messages, model, temperature):
        try:
            response = self.gpt_openai(app_id, api_key, base_url, messages, model, temperature=temperature)
            result["response"] = response[0]  # 存储响应
        except Exception as e:
            print(f"Error in gpt_openai: {e}")
            result["response"] = None

    def gpt_openai(self, app_id, api_key, base_url, messages, model, max_tokens=None, temperature=1, silent=False):
        client = openai.OpenAI(
            api_key=f'{app_id}.{api_key}',
            base_url=base_url,
        )

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            data.update({"max_tokens": max_tokens})

        if not silent:
            print("gpt data: {}".format(data))

        response = client.chat.completions.create(**data)
 
        return (f'{response.choices[0].message.content}',)