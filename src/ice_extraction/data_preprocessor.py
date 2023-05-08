import json

L_WIN = [-1, 0, 1]


def process_empty(data_raw):
    for i in range(len(data_raw)):
        data_raw[i]['text_a'] = data_raw[i]['text_a'].replace('\n', '')
    return data_raw


def process_sentences(data_raw):
    # code_classes = []
    searched_token = '[CODE'
    # for i in range(10):
    #     code_classes.append("[CODE{}]".format(i + 1))
    for i in range(len(data_raw)):
        l_text = data_raw[i]['text_a'].split('.')
        l_text_searched_record = [i_code_search for i_code_search in range(len(l_text)) if
                                  searched_token in l_text[i_code_search]]
        l_ret_sentences = []
        for index_code_list in l_text_searched_record:
            for _record_l_win in L_WIN:
                l_ret_sentences.append(index_code_list + _record_l_win)
        l_ret_sentences = list(set(l_ret_sentences))
        l_ret_sentences.sort()
        l_ret_sentences_limit = [_record_ret_sentence for _record_ret_sentence in l_ret_sentences if
                                 0 <= _record_ret_sentence < len(l_text)]
        # l_text_searched = [_record for _record in l_text if searched_token in _record]
        # text_ret = '. '.join(l_text_searched)
        text_ret = ''
        for _record_ret_sentence_limit in l_ret_sentences_limit:
            text_ret += (l_text[_record_ret_sentence_limit] + '.')
        data_raw[i]['text_a'] = text_ret
    return data_raw


def process_pipeline(data_raw):
    # data_raw = process_empty(data_raw)
    data_raw = process_sentences(data_raw)
    return data_raw


if __name__ == '__main__':
    with open('../../data/data_code_format.json', 'r', encoding='utf-8') as json_in:
        data = json.load(json_in)
        json_in.close()
    data_store = process_pipeline(data)
    print('--------Data Preprocessing Finished--------')
    with open('../../data/data_pre_code_format.json', 'w', encoding='utf-8') as json_out:
        json_out.write(json.dumps(data_store, indent=2))
        json_out.close()
