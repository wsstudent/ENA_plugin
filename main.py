from gradioui import build_ui

if __name__ == "__main__":
    # 初始化并运行 Gradio 界面
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
