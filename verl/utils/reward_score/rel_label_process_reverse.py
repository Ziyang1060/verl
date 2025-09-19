import sys
import re
import os

def check_and_extract_four_rel_scores(content: str):
    """
    检查 content 是否包含恰好 4 个 \\boxed{xx}，并提取其中的整数。

    返回:
        Tuple[bool, Optional[List[int]]]:
            - 第一个元素表示是否完全匹配格式（即是否恰好3个）。
            - 第二个元素是提取的三个整数值列表。
    """
    # 匹配所有 \boxed{整数}，支持负数
    matches = re.findall(r"\\boxed\{(-?\d+)\}", content)

    if len(matches) == 4:
        # 转换为整数列表
        try:
            extracted_values = [int(num) for num in matches]
            return 1, extracted_values
        except Exception as e:
            return 0, (-100, -100, -100, -100)
    else:
        return 0, (-100, -100, -100, -100)


def compute_score(predict_str, ground_truth):
    # print(f"{predict_str=}")
    # print(f"{ground_truth=}")
    format_acc, pred_labels = check_and_extract_four_rel_scores(predict_str)

    pre_pred = pred_labels[0]
    init_pred = pred_labels[1]
    criteria_pred = pred_labels[2]
    final_pred = pred_labels[3]
    is_consistent = pre_pred == final_pred

    pre_pred_acc = 0
    init_pred_acc = 0
    criteria_pred_acc = 0
    final_pred_acc = 0
    ground_truth = int(ground_truth)
    if pre_pred == ground_truth:
        pre_pred_acc = 1
    if init_pred == ground_truth:
        init_pred_acc = 1
    if criteria_pred == ground_truth:
        criteria_pred_acc = 1
    if final_pred == ground_truth:
        final_pred_acc = 1

    binary_pred_acc = (1 if ground_truth >= 1 else 0) == (1 if pre_pred >= 1 else 0)

    acc_reward = 2.0 if pre_pred_acc else 0.0
    binary_acc_reward = 0.5 if binary_pred_acc else 0.0

    return {
        # "score": acc_reward + binary_acc_reward,
        "score": acc_reward, # 必选
        "pred": pre_pred, # 必选
        "pred_acc": pre_pred_acc,
        "is_pre_final_consistent": is_consistent,
        "pre_pred": pre_pred,
        "init_pred": init_pred,
        "criteria_pred": criteria_pred,
        "final_pred": final_pred,
        "pre_pred_acc": pre_pred_acc,
        "init_pred_acc": init_pred_acc,
        "criteria_pred_acc": criteria_pred_acc,
        "final_pred_acc": final_pred_acc,
        "format_acc": format_acc,
        "ground_truth": ground_truth,
        "consistency_acc": -1,
        "is_call_verifier": -1,
        "binary_ground_truth": 1 if ground_truth >= 1 else 0,
        "binary_pred":  1 if pre_pred >= 1 else 0,
        "binary_pred_acc": binary_pred_acc
    }


if __name__ == "__main__":
    print(compute_score("## 评分\n\\boxed{0}\n\n## 思考过程\n### 第一步：需求分析  \n- **query分类**：精准需求query。  \n  - **明确需求**：用户查询“苹果入驻小红书”的意图明确且唯一，即获取苹果科技公司（Apple Inc.）入驻小红书平台的相关信息，如官方账号动态、活动公告或新闻报道。  \n\n### 第二步：内容占比分析  \n- **分析**：  \n  - note标题为“来 了 来 了 , 苹果 全家桶 来 了”，内容为“远 在 异国他乡 的 兄弟 , 苹果 全家桶 # apple [ 话题 ] #   # apple 入驻 小红书 [ 话题 ] #”。  \n  - note主题聚焦于苹果产品（如“苹果全家桶”），但通过话题标签提及“apple入驻小红书”。  \n  - 与query直接相关的内容仅限于话题标签“# apple 入驻 小红书 [ 话题 ] #”，占整体内容比例小于10%（主要篇幅用于产品展示）。  \n  - 命中部分（“apple”和“入驻小红书”）与note主题（产品推广）一致，但关注重点在产品而非公司入驻。  \n- **结论**：提及或者占比小于10%（符合定义：占比小于10%，且命中部分与主题一致但关注重点偏移）。  \n\n### 第三步：类目匹配分析  \n- **query类目**：科技产品（来自query_extra）。  \n- **note类目**：美食,美食展示,食材展示（来自note_extra）。  \n- **分析**：query需求为科技公司动态，而note类目为美食（苹果指水果），二者类目不匹配。  \n\n### 第四步：关键词命中分析  \n- **query关键词**：苹果、入驻、小红书。  \n- **命中关键词**：  \n  - “苹果”：在标题和内容中出现（但指水果，非科技公司）。  \n  - “入驻”：在话题标签“# apple 入驻 小红书 [ 话题 ] #”中出现。  \n  - “小红书”：在话题标签“# apple 入驻 小红书 [ 话题 ] #”中出现。  \n- **未命中关键词**：无（所有关键词均被提及，但“苹果”语义不匹配）。  \n- **说明**：关键词“苹果”在query中指科技公司，而note中指水果，属于语义不匹配（类似示例“桃花” vs “山桃花”）。  \n\n### 第五步：初步相关性评分  \n- **分析**：  \n  - 需求满足程度：未满足主需（note主题为水果产品，非科技公司入驻）。  \n  - 内容占比：提及或小于10%（需求满足程度低）。  \n  - 类目：不匹配。  \n  - 关键词：全部命中，但语义偏移。  \n- **评分依据**：符合【1分】档位（满足需求程度低，内容提及或者占比小于10%）。  \n- **初步评分**：\\boxed{1}  \n\n### 第六步：特殊业务情况校准  \n- **分析**：  \n  - query含限定词“入驻小红书”，属于核心限定词（指定平台动态）。  \n  - note话题标签“# apple 入驻 小红书 [ 话题 ] #”直接命中该限定词，但note主题（苹果水果）与query核心意图（科技公司）不一致。  \n  - 适用规则：**核心限定词缺失**（note包含部分词汇如“苹果”，但遗漏核心限定词“入驻小红书”的语义关联，导致需求不满足）。  \n  - 校准规则：核心限定词缺失给予【0分】。  \n- **校准评分**：\\boxed{0}  \n\n### 第七步：最终评分  \n- **综合分析**：  \n  - 第五步初步评分为1分，但第六步因核心限定词缺失校准至0分。  \n  - 最终：query需求未满足（note主题偏移），内容占比低，类目不匹配，关键词语义不匹配，核心限定词缺失导致相关性低。  \n- **最终评分**：\\boxed{0}", "0"))
