import sys
# 备份原始 sys.path
_orig_path = sys.path.copy()
# 删除冲突目录
conflict_dir = "code/verl/verl/utils/reward_score"
sys.path = [p for p in sys.path if conflict_dir not in p]
# 导入 openai（内部会 import math, 而verl本身也有math文件，会冲突）
from openai import OpenAI
# 恢复原始 sys.path，让之后的项目模块可以正常导入
sys.path = _orig_path
import re
import os


VERIFIER_API_IP_ADDR = os.environ["VERIFIER_API_IP_ADDR"]
client = OpenAI(api_key="zzy_vllm",
                base_url=f"http://{VERIFIER_API_IP_ADDR}:8000/v1")
VERIFIER_PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{output}"""

Correct answer: {answer}

Judgement:
'''

RESPONSE_PATTERN = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'


def check_response_format(response):
    format_acc = 0
    think_cot = ""
    answer_pred = -100
    m = re.search(RESPONSE_PATTERN, response, flags=re.DOTALL | re.MULTILINE)
    try:
        if m:
            think_cot = m.group(1)
            answer_pred = int(m.group(2).strip())
            assert answer_pred in [-1, 0, 1, 2, 3]
            format_acc = 1
    except Exception as e:
        print(f"{e}=")
    return format_acc, think_cot, answer_pred


def get_verifier_judgement(think_cot, answer_pred):
    user_input = VERIFIER_PROMPT.format(
        question="请你对用户给出的搜索query和对应的note的相关性进行打分，范围为[-1,0,1,2,3]，即最低分为-1分，最高分为3分，具体细则如下：\n【3分】：满足主需，内容大量匹配；\n【2分】：满足主需，内容部分匹配；满足次需，内容大量匹配；满足次需，内容部分匹配；\n【1分】：满足需求程度低，内容提及或者占比小于10%；\n【0分】：不满足用户的需求，但query和note类目匹配或note命中了query中的关键词；\n【-1分】：不满足用户的需求，query和note类目不匹配且note没有命中query中的关键词；",
        output=think_cot,
        answer=answer_pred
    )
    try:
        response = client.chat.completions.create(
            model="xVerify-9B-C",
            messages=[
                {'role': 'system', 'content': "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."},
                {'role': 'user', 'content': user_input}],
            temperature=0.1,
            top_p=0.7,
            n=1,
            max_tokens=10
        ).to_dict()
        pred = response["choices"][0]["message"]["content"].strip().lower()
        if "incorrect" in pred:
            return False
        elif "correct" in pred:
            return True
    except Exception as e:
        print(f"{e=}")
        return False


def check_reasoning_consistency(think_cot, answer_pred):
    call_verifier = False
    if think_cot == "" or answer_pred == -100:
        return False, call_verifier

    rule_consistency = False
    verifier_consistency = False

    # 找到 think 中的最后一个数字
    think_last_number = None
    think_numbers = re.findall(r"-?\d+", think_cot[-30:])
    if think_numbers:
        think_last_number = int(think_numbers[-1])
        if think_last_number not in [-1, 0, 1, 2, 3]:
            think_last_number = None

    if isinstance(think_last_number, int):
        rule_consistency = (think_last_number == answer_pred)

    if rule_consistency == False:
        call_verifier = True
        verifier_consistency = get_verifier_judgement(think_cot, answer_pred)

    return rule_consistency | verifier_consistency, call_verifier


def compute_score(predict_str, ground_truth):
    # print(f"{predict_str=}")
    # print(f"{ground_truth=}")
    format_acc, think_cot, answer_pred = check_response_format(predict_str)
    reasoning_consistency_acc, is_call_verifier = check_reasoning_consistency(think_cot, answer_pred)
    reasoning_consistency_acc = int(reasoning_consistency_acc)
    is_call_verifier = int(is_call_verifier)

    pred_acc = 0
    ground_truth = int(ground_truth)
    if answer_pred == ground_truth:
        pred_acc = 1

    format_reward = 1.0 if format_acc else 0.0
    consistency_reward = 1.0 if reasoning_consistency_acc else 0.0
    answer_reward = 1.0 if pred_acc else 0.0

    # final_reward = consistency_reward + answer_reward + format_reward
    final_reward = answer_reward * (1 + 0.5 * consistency_reward) + 0.5 * format_reward
    return {
        "score": final_reward,
        "pred": answer_pred,
        "pred_acc": pred_acc,
        "format_acc": format_acc,
        "consistency_acc": reasoning_consistency_acc,
        "is_call_verifier": is_call_verifier,
        "ground_truth": ground_truth
    }


if __name__ == "__main__":
    print(compute_score("""<think>好的，我现在需要处理用户关于小红书平台搜索相关性的评分请求。首先，用户提供的查询是“i志愿忘记账号和密码怎么办”，而对应的笔记内容主要是关于华为荣耀手机忘记账号密码后的激活方法。我需要按照给定的六个步骤来分析和打分。\n\n第一步，分析用户的需求。这个查询属于精准需求，因为它是一个问句，明确询问如何解决忘记账号和密码的问题。用户的需求是找到解决i志愿账号密码忘记的具体方法。\n\n接下来，计算query在note中的内容占比。笔记内容多次提到华为荣耀手机忘记账号的激活步骤，例如进入fastboot模式、降级安卓系统、写入OEM数据等。然而，整个笔记没有提到“i志愿”或相关的账号密码找回流程，主要集中在华为手机的操作。因此，相关内容占比应该很低，可能在10%以下，属于提及或者占比小于10%的情况。\n\n第三步，类目匹配分析。查询的类目是“科技数码, 教育”，笔记的类目是“科技数码”。两者在科技数码部分匹配，但教育部分可能不相关。不过主要关注科技数码，所以类目部分匹配。\n\n第四步，关键词命中情况。查询中的关键词包括“i志愿”、“忘记账号”、“密码”、“怎么办”。笔记中提到了“忘记账号”和“密码”相关的操作，但核心的“i志愿”没有出现，属于未命中的关键词。\n\n第五步，初步评分。由于query是精准需求，而note仅提及部分相关关键词但未满足主需求，内容占比低，因此初步评分为1分。\n\n第六步，验证特殊业务情况。这个query属于问答类，用户提问如何解决特定问题。但note中并没有直接回答i志愿的解决方案，而是针对华为手机的操作，属于回答部分满足但相关性较低的情况，可能符合【1分】的条件。此外，没有涉及限定词、对比类、排行类等其他特殊情况，所以无需调整。</think><answer>1</answer>""", "1"))
