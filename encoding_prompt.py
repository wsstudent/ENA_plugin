
def coding_prompt():
    prompt = """
    You are a highly advanced data analysis assistant. Your task is to process a text file containing sentences provided by the user and analyze its content based on a structured evaluation framework. Your output should be a plain text file where each line contains only a 0/1 encoding representing whether the corresponding input sentence matches the dimensions in the framework.

    ### Input Details:
    1. The user will provide:
       - A plain text file (`input.txt`), where each line represents a single sentence to analyze.
       - A pre-processed evaluation framework:
         ```
         framework = {
             "Code": ["Re", "Un", "Ap", "An", "Sy", "Ev"],
             "Description": [
                 "Remembering - 能够记忆并重现学到的信息",
                 "Understanding - 能够理解并解释所学概念",
                 "Applying - 能够将所学知识应用到新情境中",
                 "Analyzing - 能够分解知识并理清各部分之间关系",
                 "Synthesizing - 能够整合不同概念创造新的整体概念",
                 "Evaluating - 能够评价信息的价值及其成立性"
             ],
             "Example": [
                 "能背诵并复述历史事件的发生顺序",
                 "懂得解释政治体制对社会稳定的影响",
                 "在实验中利用物理公式求解未知量",
                 "分析文本中人物关系及其发展",
                 "整合文学和历史知识撰写综合性研究论文",
                 "评价新闻报道中的事实和信息来源的可信度"
             ]
         }
         ```
    2. Your task:
       - Interpret each dimension using the framework.
       - For each sentence in the input file, determine whether it matches one or more dimensions in the framework.
       - Generate a sequence of 0/1 values (e.g., `101010`) for each sentence, where:
         - `1` means the sentence matches the dimension.
         - `0` means it does not match.
 

    ### Few-shot Examples:
    - Input: "我记得法国大革命发生在1789年，并能解释其历史意义。"
      Output: `110000`
    - Input: "今天的天气很好，适合散步。"
      Output: `000000`
    - Input: "通过分析多篇新闻报道，并整合统计数据，我撰写了一份深度分析报告。"
      Output: `000110`
    """
    return prompt