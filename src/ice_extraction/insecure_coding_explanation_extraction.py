from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader, PromptForGeneration
import torch
from src.ice_extraction.Config import Config as cf
import json
from src.ice_extraction.utils import generation_format, selection_format
from transformers import AdamW
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt.utils.metrics import generation_metric
from transformers.optimization import get_linear_schedule_with_warmup


def train():
    with open('../../data/data_aug_code_format.json', 'r', encoding='utf-8') as json_in:
        data = json.load(json_in)
        # print(data)
        json_in.close()

    # dataset, labels = classification_format(data)
    datasetGen = generation_format(data)
    datasetSel, datasetSelValid = selection_format(data)

    # plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    # plm, tokenizer, model_config, WrapperClass = load_plm("albert", "albert-base-v2")

    plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")
    plmGen, tokenizerGen, model_configGen, WrapperClassGen = load_plm("t5", "t5-base")

    # promptTemplate = ManualTemplate(
    #     text='{"placeholder":"text_a"} summary: {"placeholder":"text_b"}. The {"mask"} code is insecure',
    #     tokenizer=tokenizer,
    # )
    # promptTemplate = ManualTemplate(
    #     text='{"placeholder":"text_a"}. {"mask"} code is insecure',
    #     tokenizer=tokenizer,
    # )

    promptTemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer,
    text='{"placeholder":"text_a"} The insecure [CODE] is {"mask"}',
                                          using_decoder_past_key_values=True)

    promptGenTemplate = ManualTemplate(
        text='{"placeholder":"text_a"} {"special": "<eos>"} The explanation of this sentence is {"mask"}',
        tokenizer=tokenizerGen
    )

    # promptVerbalizer = ManualVerbalizer(
    #     classes=cf.classes,
    #     label_words=cf.label_words,
    #     tokenizer=tokenizer,
    # )

    promptModel = PromptForGeneration(
        template=promptTemplate,
        plm=plm, freeze_plm=False, plm_eval_mode=False
    ).cuda()

    promptModelGen = PromptForGeneration(
        template=promptGenTemplate,
        plm=plmGen, freeze_plm=False, plm_eval_mode=False
    ).cuda()

    data_loader = PromptDataLoader(
        dataset=datasetSel,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256, decoder_max_length=10,
        batch_size=cf.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
        truncate_method="head"
    )

    data_loader_valid = PromptDataLoader(
        dataset=datasetSelValid,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256, decoder_max_length=10,
        batch_size=cf.batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=True,
        truncate_method="head"
    )

    data_loaderGen = PromptDataLoader(
        dataset=datasetGen,
        tokenizer=tokenizerGen,
        template=promptGenTemplate,
        tokenizer_wrapper_class=WrapperClassGen,
        max_seq_length=256, decoder_max_length=256,
        batch_size=cf.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
        truncate_method="head"
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in promptTemplate.named_parameters() if
                       (not any(nd in n for nd in no_decay)) and p.requires_grad],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in promptTemplate.named_parameters() if
                       any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)

    tot_step = len(data_loader) * 5
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

    # optimizer = AdamW(promptModel.parameters(), lr=3e-5)

    loss_func = torch.nn.CrossEntropyLoss()
    total_cls_loss, total_gen_loss, total_loss = 0.0, 0.0, 0.0
    total_cls_loss_last, total_gen_loss_last, total_loss_last = 0.0, 0.0, 0.0
    # making zero-shot inference using pretrained MLM with prompt
    count_step = 0
    count_parameter = 1
    for epoch in range(100):
        promptModel.train()
        for step, (inputs_selection, inputs_generation) in enumerate(
                zip(data_loader, data_loaderGen)):
            # inputs = inputs.cuda()

            # Calculate the loss of code classification.
            loss_cls = promptModel(inputs_selection.cuda())
            # loss_cls = loss_func(logits, inputs_selection['label'])
            # Calculate the loss of description generation.
            loss_gen = promptModelGen(inputs_generation.cuda())

            # Combined loss.
            loss_cls.backward()
            torch.nn.utils.clip_grad_norm_(promptTemplate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # loss = loss_cls + 0.05 * loss_gen
            # loss.requires_grad_()
            # loss.backward()
            # total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            # total_gen_loss += loss_gen.item()
            count_step += 1
            # optimizer.step()
            # optimizer.zero_grad()
            if count_step % cf.steps == 0:
                torch.save(promptModel.state_dict(), "model_saved/model_parameter_step_{}.pkl".format(count_parameter))
                count_parameter += 1
                avg_cls_loss = (total_cls_loss - total_cls_loss_last) / cf.steps
                # avg_gen_loss = (total_gen_loss - total_gen_loss_last) / cf.steps
                avg_loss = (total_loss - total_loss_last) / cf.steps
                # print("Epoch {}, CLS loss {}, GEN loss {}, total loss {}".format(epoch, avg_cls_loss, avg_gen_loss, avg_loss))
                print("Epoch {}, CLS loss {}".format(epoch, avg_cls_loss))
                evaluate(promptModel, data_loader_valid)
                total_cls_loss_last, total_gen_loss_last, total_loss_last = total_cls_loss, total_gen_loss, total_loss


def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            inputs = inputs.cuda()
            _, output_sentence = prompt_model.generate(inputs, **cf.generation_arguments)
            generated_sentence.extend(output_sentence)
            groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_score", score, flush=True)
    return generated_sentence


if __name__=='__main__':
    train()