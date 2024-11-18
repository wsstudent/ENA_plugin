import gradio as gr
import pandas as pd

# 示例数据
data = {
    "Code": ["Alice", "Bob", "Charlie"],
    "Description": [24, 30, 22],
    "Example": ["New York", "Los Angeles", "Chicago"]
}
df = pd.DataFrame(data)


# 处理用户提交的表格数据
def process_table(data):
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)
    # 这里可以对数据进行进一步处理
    print("用户提交的数据：")
    print(df)
    return df


# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 表格示例")
    dataframe = gr.Dataframe(value=df, label="用户信息表", interactive=True)

    submit = gr.Button("提交")
    submit.click(process_table, inputs=[dataframe], outputs=[dataframe])

demo.launch(server_name="127.0.0.1", server_port=7869)
