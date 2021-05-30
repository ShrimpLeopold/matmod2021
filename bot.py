# source bot/bin/activate

import telebot
from telebot  import types
#import csv
import pandas as pd
#import tensorflow as tf
from tensorflow import keras
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer




bot = telebot.TeleBot('1730646593:AAGdjGrSLbqKu8tRB6ieCXuSj6JOjkPuJ3E')

tokenizer = AutoTokenizer.from_pretrained("Grossmend/rudialogpt3_medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("Grossmend/rudialogpt3_medium_based_on_gpt2")


@bot.message_handler(commands=['start', 'help'])
def command_help(message):
    keyboard = types.InlineKeyboardMarkup()
    callback_button = types.InlineKeyboardButton(text="Predict", callback_data="predict")
    keyboard.add(callback_button)
    bot.reply_to(message, "Привет, {}! Я могу предсказать, наступит ли Эль Ниньо или Ла Ниньо в течение года. \n Набери \n /predict \n А еще со мной можно поболтать, если тебе скучно".format(message.from_user.first_name))



@bot.message_handler(commands=['predict'])
def command_help(message):
    ans = pd.read_csv('ans.csv')
    ans.drop(['Unnamed: 0'], axis=1, inplace=True)
    bot.reply_to(message, "Predicted ONI for the next 12 months: \n {}!".format(ans.to_string(index=False)))

@bot.message_handler(func=lambda message: True)
def talk(message):

    #print('talk')



    def get_length_param(text: str) -> str:
        tokens_count = len(tokenizer.encode(text))
        if tokens_count <= 15:
            len_param = '1'
        elif tokens_count <= 50:
            len_param = '2'
        elif tokens_count <= 256:
            len_param = '3'
        else:
            len_param = '-'
        return len_param

    for step in range(1):
        input_user = message.text

        # encode the new user input, add parameters and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(
            f"|0|{get_length_param(input_user)}|" + input_user + tokenizer.eos_token + "|1|1|", return_tensors="pt")

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response
        chat_history_ids = model.generate(
            bot_input_ids,
            num_return_sequences=1,
            max_length=512,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.6,
            mask_token_id=tokenizer.mask_token_id,
            eos_token_id=tokenizer.eos_token_id,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            device='cpu',
        )

        # pretty print last ouput tokens from bot
        #print(
        #    f"===> RuDialoGPT: {tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)}")

        bot.reply_to(message, tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))





bot.polling()
