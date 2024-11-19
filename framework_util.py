import pandas as pd
import os
import uuid
import re
from api_util import call_api
from typing import List
import gradio as gr

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