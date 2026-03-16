"""Chat bot.

Chat with the version of gpt-2 fine-tuned
on rutgers headlines
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


tokenizer = GPT2Tokenizer.from_pretrained('src/gpt2/models')
model = GPT2LMHeadModel.from_pretrained('src/gpt2/models')
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        max_length=64,
        num_beams=5,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print("GPT-2:", generated)


