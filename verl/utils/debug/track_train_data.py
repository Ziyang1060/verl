from typing import List
import json
from verl.protocol import DataProto


def data2json(batch: DataProto, steps: int, tokenizer) -> List:
    result = []
    for sample in batch:
        data = {}
        attention_mask = sample.batch['attention_mask']
        prompt = sample.batch['prompts']
        response = sample.batch['responses']
        prompt_mask = attention_mask[:prompt.shape[0]]
        response_mask = attention_mask[prompt.shape[0]:]

        data["prompt"] = tokenizer.decode(prompt[prompt_mask == 1])
        data["response"] = tokenizer.decode(response[response_mask == 1])
        data["response_tokens"] = [tokenizer.decode(token_id) for token_id in response[response_mask == 1]]
        data["logprobs"] = sample.batch['old_log_probs'][response_mask == 1].tolist()
        data["ref_logprobs"] = sample.batch['ref_log_prob'][response_mask == 1].tolist()
        data["token_rewards"] = sample.batch['advantages'][response_mask == 1].tolist()
        data['reward'] = sample.batch['token_level_rewards'][response_mask == 1][-1].tolist()
        data['step'] = steps
        result.append(data)
    return result


def write_json(data, output_file, mode='a'):
    with open(output_file, mode) as fout:
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    return


def track_batch(batch, filename, tokenizer, step=0):
    data = data2json(batch, step, tokenizer)
    for item in data:
        write_json(item, filename)

