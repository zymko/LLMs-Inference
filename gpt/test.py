from gpt import Text
from datasets import load_dataset
import csv

# load_dataset("wikipedia", "20220301.en")
# dataset = load_dataset("wikitext", "wikitext-103-v1")

# with open('dataset/input.txt', 'r', encoding="utf-8") as file:
#     _input = file.read()


# txt = "".join(dataset['train']['text']) + "".join(dataset['validation']['text']) + "".join(dataset['test']['text']) + _input
# txt = "".join(dataset['train']['text'])
# txt = "".join(dataset['validation']['text']) + "".join(dataset['test']['text']) + "".join(dataset['train']['text'][:50000])
# with open('dataset/text_split_2.txt', 'w', encoding="utf-8") as file:
#     file.write(txt)

# with open('dataset/text_split_2.txt', 'r', encoding='utf-8') as file:
#     text = file.read()
# mytext = Text(B=8, T=1024, text=text)

rows = rows = [

    ["mode_size", "batch_size", "sequence_length", "running_time", "peak_memory"]
]

with open("gpt_results/output.csv", "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(rows)