import gradio as gr
import pandas as pd
from framework_util import generate, regenerate, clear_history, reverse_last_round, display_excel
from framework_util import validate_and_process_file, table_gen, generate_file_from_dataframe
from coding_util import coding

def build_ui():
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
    return demo
