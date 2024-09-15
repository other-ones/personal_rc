from transformers import CLIPTextModel, CLIPTokenizer
# pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5'
pretrained_model_name_or_path='stabilityai/stable-diffusion-2-1-base'
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
eos_token_id=tokenizer.eos_token_id
pad_token_id=tokenizer.pad_token_id

# prompt="a picture of a dog"
# input_ids= tokenizer(
#                     [prompt],
#                     padding="max_length",
#                     truncation=True,
#                     max_length=tokenizer.model_max_length,
#                     # return_tensors="pt",
#                 ).input_ids[0]
# print(eos_token_id,'eos_token_id')
# print(pad_token_id,'pad_token_id')
# print(input_ids,type(input_ids))
word="teddy"
out = tokenizer.encode(word, add_special_tokens=False)
print(out)
print(tokenizer.decode(out[0]))