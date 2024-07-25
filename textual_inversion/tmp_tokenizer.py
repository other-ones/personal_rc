import torch
from transformers import CLIPTextModel, CLIPTokenizer
caption='a picture of <pet_dog1> on the Moon'
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
placeholder_token1=["<pet_dog1>"]
print(placeholder_token1,'placeholder_token1')
added=tokenizer.add_tokens(placeholder_token1)
placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)[0]
token_ids=tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]
print(token_ids)
decoded=tokenizer.decode(token_ids[0])
print(decoded,'decoded')
print(placeholder_token_id1,'placeholder_token_id1')
indexed=token_ids[torch.where(token_ids==placeholder_token_id1)]
indexed_decoded=tokenizer.decode(indexed)
print(indexed_decoded)