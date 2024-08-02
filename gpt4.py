import yaml
import openai
from .settings import load_settings

class GPT4_WWL:

    
    @classmethod
    def INPUT_TYPES(s):
        all_settings = load_settings()
        default_user_prompt = all_settings['example_user_prompt']
        available_apis = [a for a in all_settings['openai_compatible']]
        default_model = all_settings['openai_compatible']['default']['model']

        return {
            "required": {
                "api": (available_apis, {
                    "default": "default"
                }),
                "model": (default_model,
                          {"default": "gpt-4-1106-preview"}),
                "system_prompt": ("STRING",
                                  {
                                      "default": ("act as prompt generator, I will give you text and you describe an image that matches that text in details, "
                                                  "answer with one response only.if I input in Chinese to communicate with you, but it is crucial that your response be in English."),
                                      "multiline": True, "dynamicPrompts": False
                                  }),

                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": default_user_prompt
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
            },
            "optional": {
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 1.0, "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gpt_response",)
    FUNCTION = "generate"
    CATEGORY = "WWL"
    
    def generate(self, api, model: str, temperature: float | None = None,
                top_p: float | None = None, user_prompt="", system_prompt=""):
        try:
            system_content = system_prompt
            user_content = user_prompt

            messages = [
                {'role': 'user', 'content': user_content},
                {'role': 'system', 'content': system_content}
            ]

        
            response = self.gpt_openai(
                messages=messages,
                model=model,
                temperature=temperature,
            )
            return (response,)
        except Exception as e:
            print(f"Error occurred: {e}")
            return (system_content,)

    def gpt_openai(self, messages, model, max_tokens=None, temperature=1,  silent=False):
        settings = load_settings()
        api_settings = settings['openai_compatible']['default']

        APP_ID = api_settings['app_id']
        APP_KEY = api_settings['api_key']
        END_POINT = api_settings['base_url']


        client = openai.OpenAI(
            api_key=f'{APP_ID}.{APP_KEY}',
            base_url=END_POINT,
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

