from typing import List
import json
from verl.protocol import DataProto


def data2json(batch: DataProto, step: int, tokenizer) -> List:
    result = []
    for idx in range(len(batch.batch)):
        data = {}
        attention_mask = batch.batch['attention_mask'][idx]
        prompt = batch.batch['prompts'][idx]
        response = batch.batch['responses'][idx]

        prompt_mask = attention_mask[:prompt.shape[0]]
        response_mask = attention_mask[prompt.shape[0]:]
        data["prompt"] = tokenizer.decode(prompt[prompt_mask == 1])
        data["response"] = tokenizer.decode(response[response_mask == 1])
        data["response_tokens"] = [tokenizer.decode(token_id) for token_id in response[response_mask == 1]]
        if "rollout_log_probs" in batch.batch:
            data["vllm_rollout_log_probs"] = batch.batch['rollout_log_probs'][idx][response_mask == 1].tolist()
        if "old_log_probs" in batch.batch:
            data["logprobs"] = batch.batch['old_log_probs'][idx][response_mask == 1].tolist()
        if "ref_log_prob" in batch.batch:
            data["ref_logprobs"] = batch.batch['ref_log_prob'][idx][response_mask == 1].tolist()
        if "forward_entropys" in batch.batch:
            data["entropys"] = batch.batch['forward_entropys'][idx][response_mask == 1].tolist()
        if "advantages" in batch.batch:
            data["token_level_advantages"] = batch.batch['advantages'][idx][response_mask == 1].tolist()
            data['token_level_scores'] = batch.batch['token_level_scores'][idx][response_mask == 1].tolist()
            data['token_level_rewards'] = batch.batch['token_level_rewards'][idx][response_mask == 1].tolist()
            data["advantage"] = batch.batch['advantages'][idx][response_mask == 1].tolist()[-1]
            data['score'] = batch.batch['token_level_scores'][idx][response_mask == 1].tolist()[-1]
            data['reward'] = batch.batch['token_level_rewards'][idx][response_mask == 1].tolist()[-1]
        if "ground_truth" in batch.non_tensor_batch and "pred_acc" in batch.non_tensor_batch:
            data['ground_truth'] = int(batch.non_tensor_batch["ground_truth"][idx])
            data['pred_acc'] =  int(batch.non_tensor_batch["pred_acc"][idx])
        data['step'] = step
        result.append(data)
        # # ── DEBUG: print out the types ─────────────────────────────────────
        # import numpy as np
        # for key, val in data.items():
        #     # 1) catch NumPy scalars
        #     if isinstance(val, np.generic):
        #         print(f"[DEBUG] {key!r} is a NumPy scalar: {type(val)} (dtype={val.dtype})")
        #     # 2) catch lists & show element types
        #     elif isinstance(val, list):
        #         elem_types = {type(x) for x in val}
        #         print(f"[DEBUG] {key!r} is a list of element types: {elem_types}")
        #     # 3) anything else
        #     else:
        #         print(f"[DEBUG] {key!r}: {type(val)}")
        # print("—" * 40)
        # break
    return result


def write_json(data, output_file, mode='a'):
    with open(output_file, mode) as fout:
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    return


def track_batch(batch, filename, tokenizer, step=0):
    data = data2json(batch, step, tokenizer)
    for item in data:
        write_json(item, filename)