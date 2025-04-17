# a minimal iterable dataset that has fixed content but only returns iterableDataset. 

# My threads: https://discuss.huggingface.co/t/using-an-iterabledataset-for-1-epochs-in-trainer/133891/2

from typing import List, Dict, Tuple, Literal, Any, TypedDict
import json, os, time

from tqdm import tqdm
import numpy as np
import torch 
import datasets
import torch.utils # HF's 

# from torchdata.datapipes.iter import IterableWrapper # version <=0.9

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

class DataStreamer(torch.utils.data.IterableDataset):
    def __init__(self):
        # example data 
        self.ds = datasets.Dataset.from_dict(
            {
                "text1":["A", "B", "C", "D", "E"], 
                "label":[0, 0, 0, 1, 1]
                }
            ) 
        
        # Manually tokenize the dataset
        # self.ds = self.ds.map(lambda x: tokenizer(x["text1"], truncation=True, padding="max_length", max_length=20, return_tensors="pt"), batched=True)
        # self.ds = self.ds.remove_columns(["text1"]) # remove the original text column
        
        # iterator control 
        self.epoch_finish = False
        self.sample_pointer = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        while not self.epoch_finish: 
            sample = self.ds.shuffle()[self.sample_pointer]
            # sample = self.ds[self.sample_pointer] # to be yielded
            self.sample_pointer += 1
            if self.sample_pointer == len(self.ds): # dataset exhausted
                self.epoch_finish = True
            yield sample

    def __len__(self):
        return len(self.ds)

class Course(TypedDict):
    ds_name: str
    amount: int | None # For test datasets, amount == None
    # For training datasets, amount == None was initially planned to load all but specified size. But now it is irrelevant because we specify the actual amount for each dataset in a curriculum. 
    split: Literal["train", "validation", "test"]

def load_ds_for_course(course: Course):
    split = course.get("split", "train")

    ds_path = os.path.join(
        # "/data/forrest/cache/processed_datasets", 
        # f"{course['ds_name']}_{split}.jsonl",
        "./",
        f"{course['ds_name']}.jsonl",
    )
    ds = datasets.load_dataset("json", data_files=ds_path, split="train").shuffle()

    ds = ds.add_column("ds_name", [course["ds_name"]] * len(ds))

    if course["amount"] != None:
        ds_amount = min(course["amount"], len(ds))
        ds = ds.select(range(ds_amount)) # not really need given that our Streamer will stop iterating on a dataset after the ds_amount or the full dataset is exhausted, whichever comes first, but just in case
    else:
        ds_amount = len(ds)

    return ds, ds_amount

class NonCurriculumFeeder(datasets.Dataset):
    """Load all training sets once and mix them together. 
    """
    def __init__(self, curriculum: List[Course]):

        dss = []
        for course in curriculum:
            dss.append(load_ds_for_course(course)[0])

        ds = datasets.concatenate_datasets(dss)
        ds = ds.shuffle()

        self.ds = ds

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.ds)


"""Notes 

Issues: 
1. Either IterableDataset or Dataset, __len__ method is required for scheduler. 
2. For IterableDataset, even with __len__ method, the Trainer does not know when is the end of the epoch.

Todo: 
1. Ask online, how to let Trainer know when the epoch ends for IterableDataset and start over.  
   Pytorch-lightning has a solution https://github.com/Lightning-AI/pytorch-lightning/issues/2150 
   But it is not in transformers.Trainer. 
   Another solution is to use only one epoch but start inside the __iter__ method. But this does not allow me to monitor epoch-wise metrics. 
"how to use iterabledataset with trainer for more than one epoch"
2. To verify whether 2 holds for Dataset (which uses get_item(idx) to find elements), test returning sum(curriculum.values()) in __len__ method.

Now todo-1 is solved. https://discuss.huggingface.co/t/using-an-iterabledataset-for-1-epochs-in-trainer/133891

Two choices here: 
1. Use IterableDataset. But this requires torchdata.datapipes.iter import IterWrapper
2. Use Dataset. 


Todo: 
1. First debug iter_ds.py. Why iter_more_than_epoch.py workes but iter_ds.py does not work. Understanding how to port from torchdata 0.9 to 0.10.
2. Use Dataset. 

"""


class SmartCollator(DataCollatorWithPadding): 
    """Tokenize each batch on the fly"""
    def __inti__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]):
        texts = [f["text1"] for f in features]
        labels = [f["label"] for f in features]

        print (texts)
        print (labels)

        encodings = self.tokenizer(texts, truncation=True, padding="max_length", max_length=20, return_tensors="pt")

        encodings.update(labels= torch.tensor(labels))

        return encodings

# TODO: 1. use map-style dataset 2. use set_transfrom https://x.com/lhoestq/status/1361695431923806215

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=10,
    per_device_train_batch_size=2,
    report_to="none", 
    logging_steps=1,
    remove_unused_columns=False
)

curriculum = [
    {"ds_name":"ds1", "amount":10, "split":"train"},
    {"ds_name":"ds2", "amount":20, "split":"train"},
]

trainer = Trainer(
    model=model, 
    args=training_args,
    # data_collator= DataCollatorWithPadding(tokenizer=tokenizer),
    data_collator=SmartCollator(tokenizer=tokenizer),
    # train_dataset=datasets.Dataset.from_dict({"text1":["A", "B", "C", "D", "E"], "label":[0, 0, 0, 1, 1]})
    # train_dataset=DataStreamer().with_format("torch")
    # train_dataset=IterableWrapper(DataStreamer())
    train_dataset=NonCurriculumFeeder(curriculum)
)


trainer.train()