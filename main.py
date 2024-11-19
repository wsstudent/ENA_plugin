import http.client
import json
import os
import re
import uuid
from typing import List, Dict, Any
from encoding_prompt import coding_prompt
import gradio as gr
import pandas as pd


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
#纯纯的冰啊,属于是

# 初始化 API 配置
api_config = APIConfig(
    base_url="api.chatanywhere.tech",
    endpoint="/v1/chat/completions",
    api_key="sk-nb6TMYvsD6KjllF5RC3i3Xl9tokeOe7rTdNMGnSmpI8ieog1",  # 替换为您的 API 密钥
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


def generate(chat_history: List, query: str):
    # 如果用户输入为空，则不调用 API
    if not query.strip():
        return gr.update(value=""), chat_history

    print("**********聊天记录log***********\n", chat_history, end="\n\n")

    """
    
        构造系统提示词，用于提示用户当前对话的角色。
        
    """
    messages = [{"role": "system",
                 "content": "你是一个教育方向认知理论的专家，你的任务是帮助用户一步一步构建一个认知理论架构框架，下面用户将会提交关于自己理论架构的描述"}]

    for q, a in chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})
    print("**********输入log***********\n", messages, end="\n\n")

    # 调用 API
    answer = call_api(messages)

    # 更新聊天记录
    chat_history.append([query, answer])
    return gr.update(value=""), chat_history


def regenerate(chat_history: List):
    """重新生成最后一轮回答"""
    if not chat_history:
        return gr.update(value=""), chat_history

    last_query = chat_history[-1][0]
    chat_history.pop()
    return generate(chat_history, last_query)


def clear_history():
    """清空聊天记录"""
    return []


def reverse_last_round(chat_history):
    """撤回上一轮聊天记录"""
    if chat_history:
        chat_history.pop()
    return chat_history


def validate_and_process_file(file):
    try:
        df = pd.read_excel(file.name)
        required_columns = {"Code", "Description", "Example"}
        if not required_columns.issubset(df.columns):
            raise ValueError("上传的 Excel 文件不符合要求，缺少必要的列：Code, Description, Example")

        # 打印内容调试
        print("上传的表格内容:", df.to_dict(orient="list"))
        return df
    except Exception as e:
        return f"文件处理失败: {str(e)}"


def generate_file_from_dataframe(data):
    """将 DataFrame 保存为 Excel 文件"""
    print(f"当前的表格内容: {data}")  # 打印表格内容，用于调试

    # 确保 `data` 是一个字典，并对其处理
    if isinstance(data, dict):
        data = normalize_column_lengths(data)  # 确保列长度一致
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError(f"输入的数据类型不支持: {type(data)}")

    # 获取当前工作目录
    current_dir = os.path.join(os.getcwd(), "theory_framework_save")
    # 如果文件夹不存在，则创建它
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    # 生成唯一的文件名，避免冲突
    file_path = os.path.join(current_dir, f"theory_framework_{uuid.uuid4().hex}.xlsx")
    df.to_excel(file_path, index=False)
    print(f"文件已保存到: {file_path}")  # 打印文件路径，用于确认文件生成位置
    return file_path


def normalize_column_lengths(data):
    """调整字典中所有列的长度一致"""
    if not isinstance(data, dict):
        raise ValueError("输入数据必须是字典类型")
    max_len = max(len(col) for col in data.values()) if data else 0
    for key in data.keys():
        data[key] += [None] * (max_len - len(data[key]))  # 用 None 填充
    return data


# 处理表格生成
def table_gen(chat_history):
    messages = [{"role": "system",
                 "content": "你是一个教育方向认知理论的专家，你的任务是帮助用户一步一步构建一个认知理论架构框架，下面用户将会提交关于自己理论架构的描述"}]

    for q, a in chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": '能把它变成一个表标头是Code，Description，Example的表吗，其中code要尽可能用简短的编码来表示，'
                                                'Description要精选详细的介绍，Example是一个有代表性的例子，这个例子不能太短让人不会理解。最后表的输出格式为：'
                                                '# 示例数据data = { "Code": ["Alice", "Bob", "Charlie"], "Description": [24, 30, 22], "Example": ["New York", "Los Angeles", "Chicago"] } '})
    print("**********输入log***********\n", messages, end="\n\n")

    gpt_genTable = call_api(messages)
    print("**********输出log***********\n", gpt_genTable, end="\n\n")
    # 正则表达式匹配
    pattern = r"data\s*=\s*(\{.*?\})"
    match = re.search(pattern, gpt_genTable, re.DOTALL)  # re.DOTALL 允许匹配换行符

    if not match:
        raise ValueError("没有找到匹配的 data 结构")

    data_str = match.group(1)  # 只获取字典部分
    print("生成的理论框架:", data_str)

    try:
        df = pd.DataFrame(eval(data_str))
    except Exception as e:
        raise ValueError(f"DataFrame 生成失败: {str(e)}")

    return df


def list_frameworks(save_path):
    if not save_path.strip():
        return gr.update(value="错误: 保存路径不能为空", visible=True)
    if not os.path.exists(save_path):
        return gr.update(value="错误: 保存路径不存在", visible=True)
    files = [f for f in os.listdir(save_path) if f.endswith('.xlsx')]
    return gr.update(choices=files, visible=True)


def load_framework(save_path, selected_framework):
    if not save_path.strip():
        return gr.update(value="错误: 保存路径不能为空", visible=True)
    file_path = os.path.join(save_path, selected_framework)
    df = pd.read_excel(file_path)
    return df


# 展示上传的excel文件前10行
def display_excel(excel):
    df = pd.read_excel(excel)
    # 只显示前10行
    return df.head(10)


def extract_coding_results(gpt_output, chunk_size, code_length):
    # 初始化结果和警告
    extracted_lines = []
    warnings = []

    # 清理多余的换行符和包裹符号
    cleaned_output = re.sub(r'```|^\s+|\s+$', '', gpt_output, flags=re.MULTILINE).strip()

    # 1. 提取纯数字行
    numeric_lines = re.findall(r'^\d+$', cleaned_output, re.MULTILINE)
    extracted_lines.extend(numeric_lines)

    print("提取的编码行:", extracted_lines)
    # 2. 检查提取结果
    if len(extracted_lines) != chunk_size:
        warnings.append(f"提取的编码数量 ({len(extracted_lines)}) 与预期 ({chunk_size}) 不符。")

    if any(len(line) != code_length for line in extracted_lines):
        warnings.append(f"某些编码长度不符合预期: {extracted_lines}")

    # 返回提取结果和警告信息
    return extracted_lines, warnings


# 对提交的提交execel文件进行编码
def coding(th_framework, excel):
    print("\n***************coding函数***************\n")
    # 将 th_framework 转换为 data 字典
    framework = {
        "Code": th_framework["Code"].tolist(),
        "Description": th_framework["Description"].tolist(),
        "Example": th_framework["Example"].tolist()
    }
    # 将 data 字典转换为字符串
    framework_str = json.dumps(framework, ensure_ascii=False)
    # 打印转换后的字符串
    print("理论框架内容：", framework_str, sep="\n")

    # 读取 CSV 文件
    try:
        df = pd.read_excel(excel.name)
    except Exception as e:
        return f"无法读取文件，请检查文件格式或内容。\n错误: {e}"

    # 检查是否存在 'text' 列
    if 'text' not in df.columns:
        return "数据表中没有找到 'text' 列，请检查文件内容。"

    # 检测空值或无效数据
    invalid_rows = df['text'].isna() | df['text'].str.strip().eq("")
    if invalid_rows.any():
        return f"检测到无效数据！第 {list(invalid_rows[invalid_rows].index + 1)} 行为空或无效。请清理数据后再上传。"

    # 如果数据有效，处理并返回结果
    text_column = df['text']
    numbered_text = [f"{idx + 1}: {text}" for idx, text in enumerate(text_column)]

    code_length = len(th_framework["Code"])  # 编码长度
    chunk_size = 10  # 每 chunk_size 行分组
    code_columns = th_framework["Code"]  # 将编码内容拆分为独立的列
    # 先添加列名到 df
    for col in code_columns:
        df[col] = None
    print("要进行编码的表", df.head(10), sep="\n")

    string_array = [
        "\n".join(numbered_text[i:i + chunk_size])
        for i in range(0, len(numbered_text), chunk_size)
    ]

    prompt = coding_prompt()
    for idx, chunk in enumerate(string_array):
        # 动态确定实际的分组大小
        actual_size = min(chunk_size, len(df) - idx * chunk_size)

        # 每次重新初始化 GPT 消息
        messages = [{"role": "system",
                     "content": (
                         f"{prompt}请按照如下要求生成编码：\n"
                         f"1. 每行仅包含 {code_length} 位数字编码，不添加多余字符。\n"
                         f"2. 输出总行数应为 {actual_size}。\n"
                         f"3. 确保编码格式一致，例如：\n"
                         f"   010000\n"
                         f"   000000\n"
                         f"（注意：不要在编码前后添加编号或其他符号）"
                     )}]
        user_message = f"text 文本的内容:\n{chunk}\n理论框架是 framework={framework_str}"
        messages.append({"role": "user", "content": user_message})
        print("**********输入 log ***********\n", messages, end="\n\n")

        try:
            # 调用 GPT 接口
            gpt_gen = call_api(messages)
            print("**********输出 log ***********\n", gpt_gen, end="\n\n")

            # 使用提取函数获取结果和警告信息
            lines, warnings = extract_coding_results(gpt_gen, actual_size, code_length)

            # 如果提取结果有警告进入重试逻辑
            while warnings:
                print(f"警告信息: {warnings}")

                # 动态调整提示信息，加入警告
                warning_message = " ".join(warnings)
                messages.append(
                    {"role": "user", "content": f"返回的内容不符合要求。{warning_message} 请按系统要求重新生成。"})

                # 重新调用 GPT 接口
                gpt_gen = call_api(messages)
                print("**********重新调用 API 输出 log ***********\n", gpt_gen, end="\n\n")

                # 重新提取编码结果和警告
                lines, warnings = extract_coding_results(gpt_gen, actual_size, code_length)

            # 确定编码范围
            start_idx = idx * chunk_size
            end_idx = min(start_idx + actual_size, len(df))

            # 将每个编码字符串拆分为字符列表
            lines_2d = [list(line) for line in lines]

            # 调试输出
            print(f"编码行索引范围: {start_idx} 到 {end_idx - 1}")
            print(f"lines_2d 数据:\n{lines_2d}")
            print(f"目标列: {code_columns}")

            # 检查数据长度是否匹配
            assert len(lines_2d) == end_idx - start_idx, "编码行数与数据范围不匹配"

            # 填充 DataFrame
            for i, col in enumerate(code_columns):
                df.loc[start_idx:end_idx - 1, col] = [row[i] for row in lines_2d[:end_idx - start_idx]]

            # 打印更新后的 DataFrame 的最后几行
            print("更新后的 DataFrame 的最后几行:\n", df.iloc[end_idx - actual_size:end_idx])

        except Exception as e:
            print(f"调用 API 失败，请检查接口或输入。\n错误: {e}")
            break

    # 保存编码后的数据
    # 获取当前工作目录
    current_dir = os.path.join(os.getcwd(), "coded_excel_save")
    # 如果文件夹不存在，则创建它
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    # 生成唯一的文件名，避免冲突
    excel_path = os.path.join(current_dir, f"coded_excel_{uuid.uuid4().hex}.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"文件已保存到: {excel_path}")  # 打印文件路径，用于确认文件生成位置
    return excel_path


# Gradio UI
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 理论框架生成")
    with gr.Row():
        with gr.Column(scale=2, min_width=300):
            chatbot = gr.Chatbot(label="聊天窗口", height=400)
        with gr.Column(scale=2, min_width=300):
            user_input = gr.Textbox(label="用户输入", placeholder="请输入有关你的认知理论架构的描述...", lines=3)
            with gr.Row():
                submit = gr.Button("发送")
                clear = gr.Button("清除记录")
                regen = gr.Button("重新生成")
                reverse = gr.Button("撤回")
                tableGen = gr.Button("生成理论框架")

    gr.Markdown("## 理论框架\n\n你还可以编辑表格来自定义你的理论框架。")
    with gr.Row():
        with gr.Column(scale=5):
            # 示例数据
            data = {
                "Code": [],
                "Description": [],
                "Example": []
            }

            TH_table = pd.DataFrame(data)
            # 创建一个 Dataframe 组件存储理论框架
            dataframe = gr.Dataframe(value=TH_table, label="理论框架", interactive=False,
                                     col_count=(len(TH_table.columns), 'fixed'))
        with gr.Column(scale=1):
            updatePath = gr.File(label="上传理论框架", file_types=[".xlsx"])
            updatePath.change(validate_and_process_file, inputs=[updatePath], outputs=[dataframe])
            SaveTable = gr.Button("导出理论框架")
            print(dataframe.value)  # 调试输出，查看 `dataframe.value` 的实际内容
            # 修改 SaveTable 按钮的绑定逻辑
            SaveTable.click(
                generate_file_from_dataframe,
                inputs=[dataframe],
                outputs=gr.File(label="下载理论框架")
            )

    # 绑定按钮事件

    submit.click(
        generate,
        inputs=[chatbot, user_input],
        outputs=[user_input, chatbot]
    )
    regen.click(
        regenerate,
        inputs=[chatbot],
        outputs=[user_input, chatbot]
    )
    clear.click(clear_history, inputs=[], outputs=[chatbot])
    reverse.click(reverse_last_round, inputs=[chatbot], outputs=[chatbot])
    # 根据用户与gpt的对话生成表格
    tableGen.click(
        table_gen,
        inputs=[chatbot],
        outputs=[dataframe]
    )

    gr.Markdown("## 用理论框架进行编码")
    with gr.Row():
        file_input = gr.File(label="上传 Excel 文件", file_types=[".xlsx"])  #要编码的文件
        output = gr.Dataframe(label="前10行数据")
    with gr.Row():
        startCode = gr.Button("开始编码")

    file_input.change(display_excel, inputs=[file_input], outputs=[output])
    startCode.click(
        coding,
        inputs=[dataframe, file_input],
        outputs=gr.File(label="下载编码好的excel")
    )

demo.queue()
demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
