import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import os
#my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_device = torch.device('cpu')
kl_divergence = torch.nn.KLDivLoss()
from transformers import StoppingCriteria, StoppingCriteriaList
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
      super().__init__()
      self.stops = stops
      self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      stop_count = 0
      for stop in self.stops:
        stop_count = (stop == input_ids[0]).sum().item()

      if stop_count >= self.ENCOUNTERS:
          return True
      return False
def get_output_scores(scores, sequences):
    scores_selected = []
    for i in range(sequences.shape[-1]):
        score = scores[i]
        score = torch.softmax(score, dim=-1)
        score = score[:, sequences[:, i]]
        scores_selected.append(score)
    return scores_selected


def mask_key_words(input_ids, target_words_group, tokenizer):
    target_words = []
    for target_word in target_words_group:
        target_words += list(set([target_word, target_word.lower(), target_word.upper(), target_word.capitalize()]))
        target_words = target_words + [' '+x for x in target_words]
    target_words += tokenizer.all_special_tokens
    target_input_ids = [tokenizer(x, return_tensors="pt", add_special_tokens=False).input_ids.to(my_device) for x in target_words]
    mask = torch.ones_like(input_ids)
    for target_id in target_input_ids:
        for i in range(input_ids.shape[-1]-target_id.shape[-1]):
            if (input_ids[:, i:i+target_id.shape[-1]] == target_id).min():
                mask[:, i:i+target_id.shape[-1]] = 0
    return mask

def mask_input_ids(input_ids_ori, target_words_group, tokenizer, unk_token_id=0):
    target_words = []
    input_ids = input_ids_ori.clone().to(my_device)
    for target_word in target_words_group:
        target_words += list(set([target_word, target_word.lower(), target_word.upper(), target_word.capitalize()]))
        target_words = target_words + [' '+x for x in target_words]
    target_words = sorted(target_words, key=lambda x:-len(x))
    target_input_ids = [tokenizer(x, return_tensors="pt", add_special_tokens=False).input_ids.to(my_device) for x in target_words]
    for target_id in target_input_ids:
        for i in range(input_ids.shape[-1]-target_id.shape[-1]):
            if (input_ids[:, i:i+target_id.shape[-1]] == target_id).min():
                #change the matched input_ids to <pad> id
                input_ids[:, i:i+target_id.shape[-1]] = unk_token_id
    return input_ids

def mask_input_words(input_text, target_words_group, tokenizer):
    target_words = []
    for target_word in target_words_group:
        target_words += list(set([target_word, target_word.lower(), target_word.upper(), target_word.capitalize()]))
    target_words = sorted(target_words, key=lambda x:-len(x))
    for target_word in target_words:
        input_text = input_text.replace(target_word, tokenizer.unk_token)
    return input_text

def isalphadigit(word):
    return word.isalpha() or word.isdigit()

def mask_stop_words(input_ids, tokenizer):
    words = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=False)
    mask = torch.ones_like(input_ids).to(my_device)
    for i in range(input_ids.shape[-1]):
        tword = ''.join(filter(isalphadigit, words[i].replace('<0x0A>', '').replace('Ġ', '')))
        if tword in stopwords or tword.lower() in stopwords or len(tword)==0:
            #print(words[i])
            mask[:, i] = 0
    return mask

def unk_key_words(input_sentence, target_word, unk_word):
    return input_sentence.replace(target_word, unk_word)

def ask_original_question(question, model, tokenizer, stopping_criteria, max_length=100):
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(my_device)
    outputs = model.generate(input_ids, output_scores=True, return_dict_in_generate=True, max_length=max_length+input_ids.shape[-1], stopping_criteria=stopping_criteria)
    print('------------------------------------------------')
    print("Ori Question Output:\n{}".format(tokenizer.decode(outputs.sequences[0])))
    outputs.sequences = outputs.sequences[:, input_ids.shape[-1]:]
    original_decoding_scores = get_output_scores(outputs.scores, outputs.sequences)
    #print("Ori Question Output Sub Tokens:{}".format(tokenizer.convert_ids_to_tokens(outputs.sequences[0])))
    return outputs, original_decoding_scores

def get_familar_scores(scores, sequences):
    scores_selected = []
    for i in range(sequences.shape[-1]):
        score = scores[i]
        score = torch.softmax(score, dim=-1)
        std = torch.std(score, dim=-1)
        score = score/torch.exp(std)
        score = score[:, sequences[:, i]]
        scores_selected.append(score)
    return scores_selected

def get_mask_score(question, mask_target, all_targets, outputs, original_decoding_scores, model, tokenizer, is_full=False, stopping_criteria=None, unk_token_id=0):
    question_mask = unk_key_words(question, mask_target, tokenizer.unk_token)
    print("Masked Question:\n{}".format(question_mask))
    input_ids_mask = tokenizer(question_mask, return_tensors="pt").input_ids.to(my_device)
    weight_mask = mask_key_words(outputs.sequences, all_targets, tokenizer)
    stop_mask = mask_stop_words(outputs.sequences, tokenizer)
    mask = weight_mask * stop_mask
    tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0])
    if not is_full:
        forced_mask_ids = mask_input_ids(outputs.sequences, [mask_target], tokenizer, unk_token_id=unk_token_id)
    else:
        forced_mask_ids = mask_input_ids(outputs.sequences, all_targets, tokenizer, unk_token_id=unk_token_id)
    #print("Masked Input:{}".format(tokenizer.decode(forced_mask_ids[0], skip_special_tokens=True)))
    outputs_forced = model.generate_force(input_ids_mask, forced_mask_ids, output_scores=True,
                                          return_dict_in_generate=True, max_length=100+input_ids_mask.shape[-1], stopping_criteria=stopping_criteria)
    outputs_forced.sequences = outputs_forced.sequences[:, input_ids_mask.shape[-1]:]
    forced_decoding_scores = get_output_scores(outputs_forced.scores, outputs_forced.sequences)
    #print(tokenizer.decode(outputs_forced.sequences[0], skip_special_tokens=True))
    # compute the KL divergence between the original decoding scores and the forced decoding scores
    original_decoding_scores = torch.cat(original_decoding_scores, dim=-1)
    forced_decoding_scores = torch.cat(forced_decoding_scores, dim=-1)
    original_decoding_scores = original_decoding_scores[mask>0]
    forced_decoding_scores = forced_decoding_scores[mask>0]
    score_difference = kl_divergence(torch.log(forced_decoding_scores), original_decoding_scores) + kl_divergence(torch.log(original_decoding_scores), forced_decoding_scores)
    return score_difference.item()

def process_targets(question, targets, outputs, original_decoding_scores, model, tokenizer, stopping_criteria, unk_token_id):
    return get_mask_score(question, targets[-1], targets, outputs, original_decoding_scores, model, tokenizer, is_full=True, stopping_criteria=stopping_criteria, unk_token_id=unk_token_id)

def convert_ids_to_tokens(self, input_ids):
    rs = []
    for input_id in input_ids:
        rs.append(self.id2word[int(input_id)])
    return rs

#kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
#accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
