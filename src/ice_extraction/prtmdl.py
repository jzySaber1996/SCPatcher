from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader, PromptForGeneration
import torch
from src.ice_extraction.Config import Config as cf
from transformers import AdamW


classes = [
    "C1",
    "C2",
    "No"
]


dataset = [
    InputExample(
        guid = 0,
        text_a = "Albert Einstein was one of the greatest intellects in [CODE1].",
        label=2
    ),
    InputExample(
        guid = 1,
        text_a = "The film was not secure in [CODE1], said by [CODE2].",
        label=1
    ),
    InputExample(
        guid = 2,
        text_a = "That is wonderful [CODE1].",
        label=2
    ),
]

plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

promptTemplate = ManualTemplate(
    text='{"placeholder":"text_a"} Insecure code is {"mask"}',
    tokenizer=tokenizer,
)


promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "C1": ["first code"],
        "C2": ["second code"],
        "No": ["No insecure code"]
    },
    tokenizer = tokenizer,
)


promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer
)


data_loader = PromptDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass
)


optimizer = AdamW(promptModel.parameters(), lr=3e-5)


loss_func = torch.nn.CrossEntropyLoss()
total_cls_loss, total_gen_loss, total_loss = 0.0, 0.0, 0.0
total_cls_loss_last, total_gen_loss_last, total_loss_last = 0.0, 0.0, 0.0
# making zero-shot inference using pretrained MLM with prompt
promptModel.eval()
count_step = 0
for epoch in range(20):
    with torch.no_grad():
        for step, inputs_classification in enumerate(data_loader):
            # inputs = inputs.cuda()

            # Calculate the loss of code classification.
            logits = promptModel(inputs_classification)
            loss_cls = loss_func(logits, inputs_classification['label'])

            # loss.
            loss_cls.requires_grad_()
            loss_cls.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_cls_loss += loss_cls.item()
            # total_gen_loss += loss_gen.item()
            count_step += 1
            # optimizer.step()
            # optimizer.zero_grad()
            if count_step % 5 == 0:
                avg_cls_loss = (total_cls_loss - total_cls_loss_last) / cf.steps
                # avg_gen_loss = (total_gen_loss - total_gen_loss_last) / cf.steps
                avg_loss = (total_loss - total_loss_last) / cf.steps
                # print("Epoch {}, CLS loss {}, GEN loss {}, total loss {}".format(epoch, avg_cls_loss, avg_gen_loss, avg_loss))
                print("Epoch {}, CLS loss {}".format(epoch, avg_cls_loss))

                total_cls_loss_last, total_gen_loss_last, total_loss_last = total_cls_loss, total_gen_loss, total_loss
