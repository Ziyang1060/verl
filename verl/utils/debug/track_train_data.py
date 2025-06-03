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

        data["logprobs"] = batch.batch['old_log_probs'][idx][response_mask == 1].tolist()
        data["ref_logprobs"] = batch.batch['ref_log_prob'][idx][response_mask == 1].tolist()
        data["token_rewards"] = batch.batch['advantages'][idx][response_mask == 1].tolist()
        data['score'] = batch.batch['token_level_scores'][idx][response_mask == 1][-1].tolist()
        data['reward'] = batch.batch['token_level_rewards'][idx][response_mask == 1][-1].tolist()
        data['step'] = step
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