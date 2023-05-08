import json
from collections import Counter
import random
from src.ice_extraction.Config import Config as cf


def aug_replace(data_raw_aug):
    data_raw_aug_ret = {'guid': data_raw_aug['guid'], 'tgt_text': data_raw_aug['tgt_text'],
                        'label': data_raw_aug['label'], 'code': data_raw_aug['code']}
    text_a = data_raw_aug['text_a']
    text_a_ret = text_a.replace('\n', '')
    l_text_a = text_a_ret.split(' ')
    for each_aug in range(cf.aug_times):
        rd_replace = random.random()
        if rd_replace > cf.PROB:
            continue
        start_index, end_index = random.randint(0, len(l_text_a) - 1), random.randint(0, len(l_text_a) - 1)
        # while code_detect(l_text_a[start_index]) and code_detect(l_text_a[end_index]):
        #     start_index, end_index = random.randint(0, len(l_text_a) - 1), random.randint(0, len(l_text_a) - 1)
        temp_data = l_text_a[start_index]
        l_text_a[start_index] = l_text_a[end_index]
        l_text_a[end_index] = temp_data
    text_a_ret = ' '.join(l_text_a)
    data_raw_aug_ret['text_a'] = text_a_ret
    return data_raw_aug_ret


def aug_delete(data_raw_delete):
    data_raw_delete_ret = {'guid': data_raw_delete['guid'], 'tgt_text': data_raw_delete['tgt_text'],
                           'label': data_raw_delete['label'], 'code': data_raw_delete['code']}
    text_a = data_raw_delete['text_a']
    text_a_ret = text_a.replace('\n', '')
    l_text_a = text_a_ret.split(' ')
    for delete_index in range(len(l_text_a)):
        rd_delete = random.random()
        if rd_delete > cf.PROB or '[CODE' in l_text_a[delete_index]:
            continue
        l_text_a[delete_index] = ''
    text_a_ret = ' '.join(l_text_a)
    data_raw_delete_ret['text_a'] = text_a_ret
    return data_raw_delete_ret


def aug_insert(data_raw_insert):
    data_raw_insert_ret = {'guid': data_raw_insert['guid'], 'tgt_text': data_raw_insert['tgt_text'],
                           'label': data_raw_insert['label'], 'code': data_raw_insert['code']}
    text_a = data_raw_insert['text_a']
    text_a_ret = text_a.replace('\n', '')
    l_text_a = text_a_ret.split(' ')
    for insert_index in range(len(l_text_a)):
        rd_insert = random.random()
        if rd_insert > cf.PROB:
            continue
        word_index = random.randint(0, len(l_text_a) - 1)
        l_text_a[insert_index] += (' ' + l_text_a[word_index])
    text_a_ret = ' '.join(l_text_a)
    data_raw_insert_ret['text_a'] = text_a_ret
    return data_raw_insert_ret


def pipeline(data_raw):
    data_raw = aug_replace(data_raw)
    data_raw = aug_delete(data_raw)
    data_raw = aug_insert(data_raw)
    return data_raw


if __name__ == '__main__':
    with open('../../data/data_pre_code_format.json', 'r', encoding='utf-8') as json_in:
        data = json.load(json_in)
        json_in.close()
        # print(data)
    class_labels = [_record['label'] for _record in data]
    count_res = Counter(class_labels)
    count_res_rate = {}
    label_number_max = max(count_res.values())
    for each_class in count_res.keys():
        count_res_rate[each_class] = int(label_number_max / count_res[each_class])
    data_aug = []
    for aug_times in range(cf.aug_times):
        for _record in data:
            # data_aug.append(_record)
            aug_max = count_res_rate[_record['label']]
            for i_aug in range(aug_max):
                # print('ttt')
                data_aug.append(pipeline(_record))
    for i in range(len(data_aug)):
        data_aug[i]['guid'] = i
    print('--------Data Augmentation Finished--------')
    with open('../../data/data_aug_code_format.json', 'w', encoding='utf-8') as json_out:
        json_out.write(json.dumps(data_aug, indent=2))