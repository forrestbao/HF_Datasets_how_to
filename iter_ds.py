# Curriculum Learning with Torch IterableDataset

# Curriculum learning is powerful, but most `torch`/`transformers` tutorials online assume a fixed sequential training set, which differs from curriculum learning that traverse datasets repetitively. 

# One brutal force solution is to create a "multilayer sandwich" of data -- patty from one dataset, sesame bun from another, cheese from the 3rd -- and feed that as a huge training set to your training framework. This approach consumes a lot of RAM. In Vectara's case, we need to pre-sample multi-/cross-lingual data. 

# There is a discussion on the Hugging Face forum (https://discuss.huggingface.co/t/can-trainer-be-customised-for-curriculum-learning/22173) but no answer. 

# Instead, Vectara wants to pair multi-/cross-lingual data on-the-fly, load a dataset only when it is its turn  and repeat it so for all datasets in all epochs. 

# The example here assumes a curriculum consisting of two training sets, stored in two separate JSON files. At any time, only one dataset is loaded into the RAM.

# Dataset 1:
# ds1.json
# {"text": ["A", "B", "C", "D", "E"], "label": [0, 0, 0, 0, 0]}
# Dataset 2:
# ds2.json
# {"text": ["---X", "---Y", "---Z"], "label": [1, 1, 1]}

# A curriculum is a dictionary where the key is the dataset name and the value is the number of samples to use from that dataset in each epoch, for example: 
# curriculum = {"ds1": 4, "ds2": 2}

from typing import List, Dict, Tuple, Literal, Any
import time
import json

from tqdm import tqdm
import numpy as np
import torch
import datasets
import peft 
import torch.utils # HF's 
from torchdata.datapipes.iter import IterableWrapper # version <=0.9

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding

num_proc = 40

def load_ds_from_name(ds_name: str, task: Literal["cls", "gen"]):
    """The labels in our datasets are integers by default. It needs to be turned into strings for generation formulization. 
    """

    # TODO: add spliti
    # TODO: support load from JSONL

    ds = datasets.Dataset.from_dict(json.load(open(f"{ds_name}.json"))).shuffle()
    if task == "gen":
        ds = ds.map(lambda x: {"label": str(int(x["label"]))}, num_proc=40)
    return ds
class CurriculumStreamer(torch.utils.data.IterableDataset):
    def __init__(self, 
                curriculum: Dict[str, int], 
                ):
        """

        An example of `curriculum` is {"ds1": 4, "ds2": 2} which means first use 4 samples from ds1 and then 2 samples from ds2 in each epoch.
        """

        self.curriculum = curriculum
        self.dataset_pointer = 0
        self.sample_pointer = 0
        self.epoch_finish = False 

        self.ds_name = list(self.curriculum.keys())[self.dataset_pointer]
        self.ds_amount = self.curriculum[self.ds_name]
        self.ds = datasets.Dataset.from_dict(json.load(open(f"{self.ds_name}.json"))).shuffle()

    def __iter__(self):
        while not self.epoch_finish: 
            sample = self.ds[self.sample_pointer] # to be yielded

            if self.sample_pointer == self.ds_amount - 1: # current dataset is exhausted
                # move to the next dataset, or set self.epoch_finish to True if the last dataset is reached
                if self.dataset_pointer == len(self.curriculum) - 1:
                    self.epoch_finish = True  # end the epoch
                else:
                    self.dataset_pointer += 1

                    # load the next dataset
                    self.ds_name = list(self.curriculum.keys())[self.dataset_pointer]
                    self.ds_amount = self.curriculum[self.ds_name]
                    self.ds = datasets.Dataset.from_dict(json.load(open(f"{self.ds_name}.json"))).shuffle()
                    
                    self.sample_pointer = 0
            else: 
                self.sample_pointer += 1

            yield sample

    def __len__(self):
        return sum(self.curriculum.values())

# streamer = CurriculumStreamer(
#     curriculum={"ds1": 4, "ds2": 2}
# )

# for _ in range(15):
#     try: 
#         print(_, next(iter(streamer)))
#     except StopIteration:
#         break 

# exit()

# for epoch in range(3):
#     print(f"Epoch {epoch} ==========")
#     streamer = CurriculumStreamer(
#         curriculum={"ds1": 3, "ds2": 2}
#     )

#     train_loader = torch.utils.data.DataLoader(dataset = streamer, batch_size=2, shuffle=False, drop_last=True)

#     for i, batch in enumerate(train_loader):
#         print (f"Batch {i}")
#         print(batch)
#         time.sleep(0.2)

# exit()



class SmartCollator(DataCollatorWithPadding):
    def __init__(self, foundation: str):
        self.tokenizer = AutoTokenizer.from_pretrained(foundation)

        if "Llama-3.2-1B" in foundation : 
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]]):
        texts = [f["text1"] for f in features]
        labels = [f["label"] for f in features]

        encodings = self.tokenizer(texts, truncation=True, 
                                   padding="longest", 
                                   max_length=16000, 
                                   return_tensors="pt"
                                )

        # TODO: remove samples that are too long

        encodings.update(labels= torch.tensor(labels))

        return encodings

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).astype(np.float32).mean().item()}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=3,
    # warmup_steps=500,
    weight_decay=0.1,
    # logging_dir='./logs',
    logging_steps=1,
    report_to="none", 
    remove_unused_columns=False, 
    eval_strategy="epoch"
)

curriculum={"ds1": 2, "ds2": 2}

# foundation = "distilbert-base-uncased"
# foundation = "microsoft/Phi-3.5-mini-instruct"
foundation= "meta-llama/Llama-3.2-1B"
model = AutoModelForSequenceClassification.from_pretrained(
    foundation, 
    torch_dtype=torch.bfloat16, 
    # attn_implementation="flash_attention_2"
    )

if "Llama-3.2-1B" in foundation:
    model.config.pad_token_id = model.config.eos_token_id

peft_config = {
    "task_type": "SEQ_CLS", 
    "inference_mode": False, 
    "r": 8, 
    "lora_alpha": 32, 
    "lora_dropout" : 0.1
}

# if "Phi-3" in foundation: 
#     peft_config["target_modules"] = ["o_proj", "qkv_proj"]
#     peft_config["modules_to_save"] = ["score"]

peft_config = peft.LoraConfig(**peft_config)

model = peft.get_peft_model(model, peft_config)
model.print_trainable_parameters()

trainer = Trainer(
    model=model, 
    args=training_args,
    # data_collator= DataCollatorWithPadding(tokenizer=tokenizer),
    data_collator= SmartCollator(foundation=foundation),
    train_dataset=IterableWrapper(CurriculumStreamer(curriculum=curriculum)), 
    eval_dataset=IterableWrapper(CurriculumStreamer(curriculum=curriculum)), 
    compute_metrics=compute_metrics
)

# for batch in trainer.get_train_dataloader():
#     print ("A new batch")
#     print (batch)

trainer.train()
# trainer.evaluate()


# References:
# * https://discuss.huggingface.co/t/using-an-iterabledataset-for-1-epochs-in-trainer/133891/2