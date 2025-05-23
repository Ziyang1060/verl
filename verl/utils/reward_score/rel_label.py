import re

def compute_score(predict_str, ground_truth):
    pattern = r'.*?</think>\s*<answer>.*?</answer>'
    matches = re.match(pattern, predict_str, re.DOTALL | re.MULTILINE)
    format_acc = 1 if matches else 0
    
    assert int(ground_truth) in [-1,0,1,2,3]
    label = int(ground_truth)
    pattern = r'<answer>(-1|[0-3])</answer>'
    matches = re.findall(pattern, predict_str)
    pred_acc = 0
    pred = -100
    if len(matches) > 0 and matches[-1].strip() in ['-1','0','1','2','3']:
        pred = int(matches[-1])
        if pred == label:
            pred_acc = 1

    format_reward = 1.0 if format_acc else 0.0
    answer_reward = 1.0 if pred_acc else 0.0

    final_reward = 1 * answer_reward + 0 * format_reward
    return {
        "score": final_reward,
        "pred": pred,
        "acc": pred_acc,
        "pred_acc": pred_acc,
        "format_acc": format_acc,
    }

if __name__ == "__main__":
    print(compute_score("""<think> 噪音 </think><answer>2</answer>""", "2"))
