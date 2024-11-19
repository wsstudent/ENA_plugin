import json
import http.client
from typing import List, Dict, Any



class APIConfig:
    """API 配置类，用于存储和管理 API 的相关信息"""

    def __init__(self, base_url: str, endpoint: str, api_key: str, allowed_params: List[str]):
        self.base_url = base_url
        self.endpoint = endpoint
        self.api_key = api_key
        self.allowed_params = allowed_params

    def filter_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """过滤传递给 API 的参数，仅保留允许的字段"""
        return {k: v for k, v in params.items() if k in self.allowed_params}

# 初始化 API 配置
api_config = APIConfig(
    base_url="api.chatanywhere.tech",
    endpoint="/v1/chat/completions",
    api_key="your_api_key",  # 替换为您的 API 密钥
    allowed_params=["model", "messages"]  # 根据 API 支持的参数调整
)


def call_api(messages: List[Dict[str, Any]], **kwargs):
    """
    调用 API，根据配置动态过滤参数。

    Args:
        messages (List[Dict[str, Any]]): 聊天消息记录
        kwargs: 其他可能的参数（例如 top_p、temperature 等）

    Returns:
        str: API 返回的内容
    """
    try:
        # 构造请求 payload
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            **kwargs
        }
        # 过滤不支持的参数
        payload = api_config.filter_params(payload)

        # 调用 API
        conn = http.client.HTTPSConnection(api_config.base_url)
        headers = {
            'Authorization': f'Bearer {api_config.api_key}',
            'Content-Type': 'application/json'
        }
        conn.request("POST", api_config.endpoint, json.dumps(payload), headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        response = json.loads(data)

        # 提取回答内容
        if "choices" in response and len(response["choices"]) > 0:
            # 打印 Token 使用量
            if "usage" in response:
                usage = response["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                print(f"[Token 使用量] Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            return response["choices"][0]["message"]["content"].strip()
        else:
            return "API返回内容为空或格式错误。"
    except Exception as e:
        return f"API调用失败: {str(e)}"
