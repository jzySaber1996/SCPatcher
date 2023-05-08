import pandas as pd
import json
from src.ice_extraction.Config import Config as cf
import re

xlsx = pd.ExcelFile(r'../../data/Query_CodePractice.xlsx')
sheet1 = pd.read_excel(xlsx, 'Query_CodePractice')
sheet_selected = sheet1[~sheet1['IsBadPractice'].isnull()][
    ['Id', 'Title', 'Question', 'AcceptedAnswer', 'BadPractice', 'BadPracticeDescription']]


# print(sheet_selected)


def replace_code_token(text_raw, insecure_code):
    # code_start_count = text_raw.count('<code>')
    # code_end_count = text_raw.count('</code>')
    # print(re.findall('<code>', text_raw))
    text_ret = text_raw
    code_label = list(cf.label_words.keys())[-1]
    count_code = 1
    code_part_ret = ""
    label_replace = False
    for m, n in zip(re.finditer('<code>', text_raw), re.finditer('</code>', text_raw)):
        # print(m.start(), n.end())
        code_part = text_raw[m.start():n.end()]
        if (insecure_code != "") and (insecure_code in code_part) and (count_code <= len(cf.label_words) - 1):
            code_label = 'C{}'.format(count_code)
            code_part_ret = code_part
            label_replace = True
        else:
            code_rep = code_part.replace('<code>', '').replace('</code>', '')
            l_code_token = re.split('[,=(;\n+\-\s]', code_rep)
            if len(l_code_token) >= 5:
                label_replace = True
            else:
                label_replace = False
        if label_replace:
            text_ret = text_ret.replace(code_part, '[CODE]')
            count_code += 1
        label_replace = False
    # if code_start_count == code_end_count:
    # for i in range(min(code_start_count, code_end_count)):
    #     pos_code_start = text_raw.find('<code>', pos_code_start + 1)
    text_ret = text_ret.replace('<code>', '').replace('</code>', '')
    return text_ret, code_label, code_part_ret


def sentence_word_limitation(text):
    l_text = text.split('.')
    for i_text in range(len(l_text)):
        l_word_text = l_text[i_text].split(' ')
        if len(l_word_text) > cf.word_sentence:
            l_word_text = l_word_text[:cf.word_sentence]
        l_text[i_text] = ' '.join(l_word_text)
    if len(l_text) > cf.sentence_total:
        l_text = l_text[:cf.sentence_total]
    text_limit = '. '.join(l_text)
    return text_limit


def code_ranking(text):
    l_code_text = text.split('[CODE]')
    if len(l_code_text) <= 1:
        return text
    text_code_ret = l_code_text[0]
    count_code_rank = 1
    for i_code_text in range(len(l_code_text) - 1):
        text_code_ret += ('[CODE{}]'.format(count_code_rank) + l_code_text[i_code_text + 1])
        count_code_rank += 1
    return text_code_ret


json_final = []
for i in range(len(sheet_selected)):
    _record = sheet_selected.iloc[i]
    # text_raw = _record['Question'] + _record['AcceptedAnswer']
    insecure_code = str(_record['BadPractice'])
    code_description = str(_record['BadPracticeDescription'])
    if insecure_code == 'nan':
        insecure_code = ""
    if code_description == 'nan':
        code_description = ""
    # print(re.findall('<code>(\s+)</code>',text_raw))
    if insecure_code == "":
        text_raw = _record['Question']
    elif code_description in _record['Question']:
        text_raw = _record['Question']
    elif code_description in _record['AcceptedAnswer']:
        text_raw = _record['AcceptedAnswer']
    else:
        text_raw = _record['Question']

    text_ret, code_label, code_part = replace_code_token(text_raw, insecure_code)
    text_ret = text_ret.replace('\n', ' ')
    if '[CODE]' not in text_ret:
        continue
    code_description = code_description.replace('\n', ' ')
    tern = re.compile(r'<[^>]+>', re.S)
    text_ret = tern.sub('', text_ret)
    code_description = tern.sub('', code_description)
    # text_ret = sentence_word_limitation(text_ret)
    text_ret = code_ranking(text_ret)
    json_data = {'guid': len(json_final), 'question': _record['Question'], 'accepted_answer': _record['AcceptedAnswer'],
                 'text_a': text_ret, 'tgt_text': code_description, 'label': code_label, 'code': code_part,
                 'insecure_code': insecure_code}
    json_final.append(json_data)

print('--------Data Collection Finished--------')
with open('../../data/data_code_format.json', 'w', encoding='utf-8') as json_out:
    json_out.write(json.dumps(json_final, indent=2))
