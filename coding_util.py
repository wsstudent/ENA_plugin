import os
import re
import json
import uuid
import pandas as pd
from encoding_prompt import coding_prompt
from api_util import call_api




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

