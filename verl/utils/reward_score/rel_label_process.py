import sys
import re
import os

def check_and_extract_three_rel_scores(content: str):
    """
    检查 content 是否包含恰好 3 个 \\boxed{xx}，并提取其中的整数。

    返回:
        Tuple[bool, Optional[List[int]]]:
            - 第一个元素表示是否完全匹配格式（即是否恰好3个）。
            - 第二个元素是提取的三个整数值列表中的最后一个数。
    """
    # 匹配所有 \boxed{整数}，支持负数
    matches = re.findall(r"\\boxed\{(-?\d+)\}", content)

    if len(matches) == 3:
        # 转换为整数列表
        try:
            extracted_values = [int(num) for num in matches]
            return 1, extracted_values
        except Exception as e:
            return 0, (-100, -100, -100)
    else:
        return 0, (-100, -100, -100)


def compute_score(predict_str, ground_truth):
    # print(f"{predict_str=}")
    # print(f"{ground_truth=}")
    format_acc, pred_labels = check_and_extract_three_rel_scores(predict_str)

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

    return {
        "score": 2.0 if final_pred_acc else 0.0,
        "pred": final_pred,
        "pred_acc": final_pred_acc,
        "init_pred": init_pred,
        "criteria_pred": criteria_pred,
        "final_pred": final_pred,
        "init_pred_acc": init_pred_acc,
        "criteria_pred_acc": criteria_pred_acc,
        "final_pred_acc": final_pred_acc,
        "format_acc": format_acc,
        "ground_truth": ground_truth,
        "consistency_acc": -1,
        "is_call_verifier": -1,
    }


if __name__ == "__main__":
    print(compute_score(r"""首先，我需要处理用户提供的查询和笔记。查询是："澳洲纽康特和国版的区别"，笔记是关于纽康特氨基酸奶粉的测评，具体比较了澳版纽康特LCP和纽康特Gold。

### 第一步：分析查询背后的需求
查询是“澳洲纽康特和国版的区别”。这是一个对比类查询，询问两个版本（澳洲版和国版）的差异。查询是明确的，带有比较意图，所以我将分类为精准需求查询。用户需求是直接了解澳洲纽康特和国版纽康特之间的区别。

### 第二步：计算查询在笔记中的内容占比
查询的关键元素是“澳洲纽康特”和“国版”。笔记讨论了澳版纽康特LCP和纽康特Gold，并比较了两者。笔记没有提到“国版”；它专注于澳版LCP和Gold版本。笔记大量讨论奶粉，但“国版”部分缺失，而“澳洲纽康特”被覆盖（作为澳版LCP）。内容覆盖了澳洲纽康特部分，但忽略了国版，因此占比低于10%。分类为“提及或者占比小于10%”。

### 第三步：分析类目匹配情况
查询类目qtax是“母婴”，笔记类目ntax也是“母婴”，因此类目完全匹配。

### 第四步：关键词命中情况
查询关键词："澳洲纽康特"、"和"、"国版"、"区别"。
- 笔记命中："纽康特"、"澳洲版纽康特"、"澳版"（多次），以及"区别"相关讨论。
- 未命中："国版"完全没有提到，可能是指中国版，但笔记没有涉及。

### 第五步：初步相关性评分
需求满足程度：查询是精准需求（对比差异），但笔记未覆盖“国版”，只讨论了澳版。内容占比低（<10%），需求满足度低。评分应为1分（满足需求程度低，内容提及或占比<10%）。
- 评分：\boxed{1}

### 第六步：校准特殊业务情况
查询是“澳洲纽康特和国版的区别”，属于对比类查询（多个对比项）。笔记只讨论了澳版的两个产品（LCP和Gold），未提国版，导致对比信息不全。根据规则：[对比信息不全]得1分。校准后评分：\boxed{2}

### 第七步：最终评分
综合所有步骤：需求不完整，内容部分相关但关键元素缺失，特殊业务校准后维持1分。最终评分：\boxed{1}
""", "1"))
