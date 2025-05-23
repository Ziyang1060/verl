import sys
import json
from datasets import Dataset

dataset_path = sys.argv[1]
course_stage = 3

with open(dataset_path) as f:
    train_dataset = [json.loads(line) for line in f.readlines()]

print(len(train_dataset))

train_dataset = Dataset.from_list(train_dataset)

data_source = "relevance_RL_label"

instruction_v3 = '你是一位精通搜索引擎优化的资深算法专家，专注于小红书平台搜索相关性评估。你擅长结合平台特性进行精准的相关性标注。\n请你对用户的搜索query和平台note的相关性进行打分，你需要按步骤遵循如下思考过程：\n \n### 第一步，根据给出的query分析用户背后的需求，严格参照query，且不参考任何note中的内容\nquery按照需求可以划分为「精准需求query」和「泛需求query」，其定义和区别如下:\n1. 精准需求query:即query的意图是明确、唯一的，其往往有问句，查找方法，做法的特征\n例如:膝关节疼痛怎么办、西红柿炒鸡蛋做法\n \n2. 泛需求query:即query可能有多个意图，按照需求的主次可细分为二类:\n主要需求:用户搜索时最直接、最基本的期待，覆盖用户的共性需求。通常是用户进行搜索时脑中最先想到、最希望立即获得解答的部分\n主要需求例如:搜索“篮球”，主需是篮球规则，篮球教学等信息；搜索“苹果”，主需是水果或者苹果手机\n \n次要需求:围绕主要需求的一些附加需求、或者特定的需求。这些需求虽然不是大众用户都会有的，但对于一部分用户来说却非常重要，通常是对query的简单限定\n次要需求例如:搜索“苹果”，次需是某部电影《苹果》；搜索“灵隐寺”，次需是“灵隐寺请手串”\n \n在该步骤中，要求首先对query进行分类，对于精准需求query，需要给出其背后的明确需求；对于泛需求query，需要尽可能列出其潜在的主要和次要需求\n \n### 第二步，计算query在note中的内容占比\n你可选择的情况有：\n1. 内容大量匹配:note中对于query的有关内容占比超过80%(例如，query:欧尚  note:欧尚z6 idd23款 6.3w 二手车;欧尚;欧尚z6智电idd;汽车 欧尚z6)\n2. 内容部分匹配:note中对于query的有关内容占比在80%～10%之中(例如，query:长安欧尚  note:上汽大众朗逸2024款提供了1.5L发动机，搭载6手自一体变速箱。长安欧尚X7 plus外观设计独特，整车线条动感时尚)\n3. 提及或者占比小于10%:query的相关概念在note中仅仅被提到，提及或者占比小于10%(例如，query:长安欧尚 note:一万预算 比亚迪FO、吉利熊猫。二万预算 铃木雨燕、福克斯、嘉年华。三万预算 POLO、宝来、飞度。四万预算 Smart、起亚K3、日产道客。五万预算 新飞度、新POLO、长安欧尚X5)\n4. 内容无关:note与query的需求不匹配，也不相关(例如，query:长安欧尚 note:分享家具，轻盈且有质感的1号航号「品名」TERMINAL1，其造型设计符合人体的流畅曲线，座椅与金属支架的平衡，远看像是飘浮在半空中)\n \n### 第三步，分析query和note的类目匹配情况\n- 参照qtax和ntax分析(作为参考信息不一定准确)，判断query和note的类目是否匹配；\n \n### 第四步，分析关键词命中情况\n- note命中了query中的哪些关键词；\n- note没有命中query中的哪些关键词；\n \n### 第五步，进行初步的相关性评分\n结合上述步骤的分析结果，你可选择的分数档位有:\n【3分】：满足主需，内容大量匹配；\n【2分】：满足主需，内容部分匹配；满足次需，内容大量匹配；满足次需，内容部分匹配；\n【1分】：满足需求程度低，内容提及或者占比小于10%；\n【0分】：不满足用户的需求，但query和note类目匹配或note命中了query中的关键词；\n【-1分】：不满足用户的需求，query和note类目不匹配且note没有命中query中的关键词；\n \n### 第六步，若query符合下列某种或多种特殊业务情况，进行评分的验证和校准\n- 含限定词的query：query中存在可能影响语义信息的限定词\n    - [限定词满足]：note中的描述精确满足所有限定的要求，直接命中限定词，该情况给予【3分】；\n    - [丢失主观限定词满足]：note中出现了与query限定词语义接近的词汇，但不是原词汇本身(例如，query:温柔 note:甜美)，描述的语义有交集给予【2分】；\n    - [事实限定词丢失]:客观现实的限定词丢失,但不影响语义信息(例如，query:杭州西湖 note:西湖),给予【3分】\n    - [限定词转义]:note中描述的内容限定词有转义的情况(例如，query:矮个子穿搭 note:高个子穿搭),给予【0分】\n    - [核心限定词缺失]:note中的内容包含了query中的部分词汇,但遗漏了核心限定词(query:东来顺涮羊肉 note:涮羊肉),给予【0分】\n- 对比类query：query中给出多个对比项\n    - [对比信息不全]：query包含多个对比项，但是note中仅详细介绍单一对比项信息，该情况给予【1分】；\n- 排行类query：query中包含寻找排行、排序的需求\n    - [排序满足]：note中出现对query中三个及以上内容的排序信息，该情况给予【3分】；\n    - [排序部分满足]：note中仅出现对query中二个内容的排序信息，该情况给予【2分】；\n    - [排序不满足]：note中仅包含对query中单个内容的描述，该情况给予【1分】；\n- 问答类query：query中出现疑问或者提问\n    - [回答直接满足]：note直接给出的query答案，并且全文主题一直在围绕query进行讨论，该情况给予【3分】；\n    - [回答部分满足]：note直接给出的query答案，但与query相关的主题仅占比在10%到80%之间，该情况给予【2分】，若小于10%，该情况给予【1分】；\n    - [推理满足]：note不直接针对query给出答案，但可以通过note内容从侧面推理出答案，该情况给予【2分】；\n- 美食类query：query与美食、食物有关\n    - [食谱缺失]：query询问某种食物的食谱或大全，而note中仅包含该种食物的一种做法，该情况给予【1分】；\n    - [食材命中]：query中指向某种食材，而note介绍了使用该食材的烹饪做法，该情况给予【3分】；\n    - [食材缺失]：query中包含多种食材，但note中仅命中了其中部分，该情况给予【0分】；\n    - [做法有关联]：note与美食烹饪相关，且note中包含的食材和做法对于query虽不匹配但有一定关联性和参考意义，该情况给予【1分】；\n- 旅游类query：query中存在对旅游攻略的需求，需要从吃、玩、住、行四个角度进行分析\n    - [全角度命中]：note从三个及以上维度给出对应地点的旅游攻略，该情况给予【3分】；\n    - [部分角度命中]：note仅给出一到两个维度的对应地点旅游攻略，该情况给予【2分】；\n- 文案类query：query需求是某种风格的文案\n    - [直接满足]：note表明自身是文案，并且可以满足query的主要需求，该情况给予【3分】；\n    - [间接满足]：note未显式表明自身是是文案，但是可以被用作为相关文案，该情况给予【2分】；\n- 数字类query：query由数字组成\n    - [数字未命中]：该数字在note中没有出现，该情况给予【0分】；\n    - [数字部分命中]：该数字是note中某个数字/编号的一部分，该情况给予【2分】；\n    - [数字完全命中]：该数字在note中完整、独立出现，该情况给予【3分】；\n- POI类query：query是个POI(Point Of Interest)兴趣点，即景点、酒店和地标等\n    - [攻略满足]：note是对应POI的旅游攻略，该情况给予【3分】；\n    - [关联满足]：note是和对应POI的相关信息，该情况给予【2分】；\n- 商品类query：query是某个品类的商品\n    - [商品满足]：note中详细介绍了query中提到的商品，该情况给予【3分】；\n    - [商品间接相关]：note中并没有直接介绍query提到的商品，但是二者在用处上仍然存在关联(query:眉笔 note:眉粉)，该情况给予【0分】；\n- IP类query：query中包含明星、作品、爱豆主题\n    - [明星名字满足]：query中包含明星的名字，且note是介绍该明星的原版作品，该情况给予【2分】，若note是介绍该明星原版作品的翻唱/跳/拍作品，该情况给予【1分】；\n    - [作品满足]：note是介绍query中的原版作品，该情况给予【3分】；若note是介绍对应的翻唱/跳/拍(非原版)，该情况给予【2分】；\n- 季节穿搭类query：query需求是季节穿搭建议和教程\n    - [季节部分匹配]：query要求春秋，而note是春季或者秋季穿搭，该情况给予【3分】；query要求春夏，而note是春季或者夏季穿搭，该情况给予【2分】；query要求秋冬，而note是秋季或冬季穿搭，该情况给予【2分】；\n    - [季节不匹配]：尽管不直接匹配query的季节需求，但需要考虑实际穿搭是否可适用：query是秋季穿搭，而note是春季穿搭，考虑春秋季节穿搭比较通用，该情况给予【2分】；\n \n用户输入的query和note分别包含在<query>和</query>和<note>和</note>标签内，query和note的类目分别包含在<qtax>和</qtax>，<ntax>和</ntax>内。你的思考过程必须包含在<think>和</think>标签内，最终相关性评分（仅1个数字）包含在<answer>和</answer>内。'

def make_map_fn(split):
    def process_fn(example, idx):
        inputs = example.pop("inputs").strip()
        label = example.pop("label")
        label = int(label)
        assert label in [-1,0,1,2,3]
        user_prompt = instruction_v3 + "\n" + inputs
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            "ability": "rel_cls",
            "reward_model": {"style": "rule", "ground_truth": label},
            "extra_info": {
                "index": idx,
                "split": split,
                "course_stage": course_stage,
                "note_id": example["note_id"],
                "inputs_length": example["inputs_length"],
                "multi_acc_rate": example["multi_acc_rate"],
                "binary_acc_rate": example["binary_acc_rate"]
            },
        }
        example.pop("raw_query")
        example.pop("note")
        example.pop("note_id")
        example.pop("inputs_length")
        example.pop("q_tax")
        example.pop("n_tax")
        example.pop("multi_acc_rate")
        example.pop("binary_acc_rate")
        return data

    return process_fn

train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
print(train_dataset[0])
train_dataset.to_parquet(dataset_path.replace(".jsonl", ".train.parquet"))