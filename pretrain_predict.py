#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from importlib import import_module


key = {
    0: 'finance',
    1: 'realty',
    2: 'stocks',
    3: 'education',
    4: 'science',
    5: 'society',
    6: 'politics',
    7: 'sports',
    8: 'game',
    9: 'entertainment'
}


class Predict:
    def __init__(self, model_name='bert', dataset='THUCNews'):
        self.x = import_module('models.' + model_name)
        self.config = self.x.Config(dataset)
        self.model = self.x.Model(self.config).to('cpu')
        self.model.load_state_dict(torch.load(self.config.save_path, map_location='cpu'))

    def build_predict_text(self, text):
        token = self.config.tokenizer.tokenize(text)
        token = ['[CLS]'] + token
        seq_len = len(token)
        mask = []
        token_ids = self.config.tokenizer.convert_tokens_to_ids(token)
        pad_size = self.config.pad_size
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        ids = torch.LongTensor([token_ids])
        seq_len = torch.LongTensor([seq_len])
        mask = torch.LongTensor([mask])
        return ids, seq_len, mask

    def predict(self, query):
        # 返回预测的索引
        data = self.build_predict_text(query)
        with torch.no_grad():
            outputs = self.model(data)
            num = torch.argmax(outputs)
        return key[int(num)]


if __name__ == "__main__":
    pred = Predict('bert')
    query = "学费太贵怎么办？"
    print(pred.predict(query))
