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

VERIFIER_API_IP_ADDR = os.environ.get("VERIFIER_API_IP_ADDR", "10.204.67.35")
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

def check_reasoning_consistency(think_cot, answer_pred):
    call_verifier = False
    if think_cot == "" or answer_pred == -100:
        return False, call_verifier

    rule_consistency = False
    verifier_consistency = False

    # 找到 think 中的最后一个数字
    think_last_number = None
    think_numbers = re.findall(r"-?\d+", think_cot)
    if think_numbers:
        think_last_number = int(think_numbers[-1])
        if think_last_number not in [-1, 0, 1, 2, 3]:
            think_last_number = None

    # print(think_cot)
    # print(think_last_number)
    if isinstance(think_last_number, int):
        rule_consistency = (think_last_number == answer_pred)

    if rule_consistency == False:
        call_verifier = True
        verifier_consistency = get_verifier_judgement(think_cot, answer_pred)

    return int(rule_consistency | verifier_consistency), int(call_verifier)


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

def check_and_extract_three_rel_scores(content: str):
    """
    检查 content 是否包含恰好 3 个 \\boxed{xx}，并提取其中的整数。

    返回:
        Tuple[bool, Optional[List[int]]]:
            - 第一个元素表示是否完全匹配格式（即是否恰好3个）。
            - 第二个元素是提取的三个整数值列表。
    """
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip().replace(r"\boxed{}","")
    # 匹配所有 \boxed{整数}，支持负数
    matches = re.findall(r"\\boxed\{(.+)\}", content)

    if len(matches) == 3:
        if "我不知道" in matches[-1]:
            return 1, (-2, -2, -2)
        # 转换为整数列表
        try:
            extracted_values = [int(num) for num in matches]
            return 1, extracted_values
        except Exception as e:
            return 0, (-100, -100, -100)
    else:
        return 0, (-100, -100, -100)

def check_and_extract_three_cot(content: str):
    # 找到所有 \boxed{数字}
    matches = list(re.finditer(r"\\boxed\{(.+)\}", content))
    
    if len(matches) != 3:
        return ["","",""]
    
    segments = []
    start = 0
    for m in matches:
        text_part = content[start:m.start()]  # 去掉 boxed 本身
        boxed_value = int(m.group(1))        # 提取 boxed 里的数字
        segments.append(text_part.strip())
        start = m.end()
    
    return segments


def compute_score(predict_str, ground_truth):
    # print(f"{predict_str=}")
    # print(f"{ground_truth=}")
    predict_str = predict_str.replace(r"\boxed{}","")
    format_acc, pred_labels = check_and_extract_three_rel_scores(predict_str)
    think_cots = check_and_extract_three_cot(predict_str)
    init_consistency_acc, init_is_call_verifier = check_reasoning_consistency(think_cots[0], pred_labels[0])
    criteria_consistency_acc, criteria_is_call_verifier = check_reasoning_consistency("\n".join(think_cots[:2]), pred_labels[1])
    final_consistency_acc, final_is_call_verifier = check_reasoning_consistency("\n".join(think_cots[:3]), pred_labels[2])
    reasoning_consistency_acc = (init_consistency_acc and criteria_consistency_acc and final_consistency_acc)
    is_call_verifier = (init_is_call_verifier or criteria_is_call_verifier or final_is_call_verifier)

    init_pred = pred_labels[0]
    criteria_pred = pred_labels[1]
    final_pred = pred_labels[2]

    init_pred_acc = 0
    criteria_pred_acc = 0
    final_pred_acc = 0
    ground_truth = int(ground_truth)
    if init_pred == ground_truth:
        init_pred_acc = 1
    if criteria_pred == ground_truth:
        criteria_pred_acc = 1
    if final_pred == ground_truth:
        final_pred_acc = 1
    binary_pred_acc = (1 if ground_truth >= 1 else 0) == (1 if final_pred >= 1 else 0)

    acc_reward = 1.0 if final_pred_acc else 0.0
    if final_pred == -2:
        acc_reward = 0.5
        final_consistency_acc = 1.0
    consistency_reward = 1.0 if reasoning_consistency_acc else 0.0
    binary_acc_reward = 0.5 if binary_pred_acc else 0.0
    # final_reward = consistency_reward * acc_reward
    final_reward = acc_reward * (1 + 0.2 * init_consistency_acc + 0.2 * criteria_consistency_acc + 0.2 * final_consistency_acc)

    return {
        "score": final_reward, # 必选
        "pred": final_pred, # 必选

        "pred_acc": final_pred_acc,

        "consistency_acc": reasoning_consistency_acc,
        "init_consistency_acc": init_consistency_acc,
        "criteria_consistency_acc": criteria_consistency_acc,
        "final_consistency_acc": final_consistency_acc,
        "is_call_verifier": is_call_verifier,
        "init_is_call_verifier": init_is_call_verifier,
        "criteria_is_call_verifier": criteria_is_call_verifier,
        "final_is_call_verifier": final_is_call_verifier,
        
        "init_pred": init_pred,
        "criteria_pred": criteria_pred,
        "final_pred": final_pred,
        "init_pred_acc": init_pred_acc,
        "criteria_pred_acc": criteria_pred_acc,
        "final_pred_acc": final_pred_acc,
        "format_acc": format_acc,
        "ground_truth": ground_truth,
        "binary_ground_truth": 1 if ground_truth >= 1 else 0,
        "binary_pred":  1 if final_pred >= 1 else 0,
        "binary_pred_acc": binary_pred_acc
    }


if __name__ == "__main__":
    print(compute_score(r"""### 第一步：分析用户需求query为“宫寒有哪些症状”，属于**精准需求query**。用户意图明确且唯一：获取宫寒的具体症状列表。无次要需求。  
### 第二步：计算query在note中的内容占比 – note中仅提到“宫寒痛经：小腹凉、经期血块多、不规律”作为体寒症状的一部分，该描述占全文比例约5%。（仅一句话，且在症列表中占比例小）。其余内容涉及泡脚方法、产品推荐、心阳理论等无关信息。 – 判断：**内容占比小于10%**（仅提及）。  
### 第三步：类目匹配情况 – query类目（qtax）：无。– note类目（ntax）：无。– 分析：query为健康/症状查询，note为养生/泡脚产品推广。类目不匹配。  
### 第四步：关键词命中情况 – query关键词：宫寒、症状。– 命中关键词：note中“宫寒”被直接提及。– 未命中关键词：症状（note未用“症状”一词，但用“表现”描述症状）。– 命中关键词：宫寒。– 未命中关键词：症状（语义相关但未原词命中）。  
### 第五步：初步相关性评分 – 满足主需（获取症状），但内容占比小于10%（仅提及），符合【1分】（满足需求程度低）。– 初步评分：\boxed{1}  
### 第六步：特殊业务情况校准 – query为问答类query（疑问句“有哪些”）。– 分析：note直接给出答案（宫寒症状列表），但相关内容占比约5%，在10%以下。符合“回答部分满足”，给予【2分】。– 校准后评分：\boxed{2}  
### 第七步：最终相关性评分 – 综合评估：– 满足主需，但内容占比低（仅提及），类目不匹配。– 符合第五步初步评分逻辑。– 最终评分：\boxed{2}""", "2"))
