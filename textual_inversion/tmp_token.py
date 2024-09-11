from transformers import CLIPTokenizer, T5ForConditionalGeneration


# pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5'
pretrained_model_name_or_path='stabilityai/stable-diffusion-2-1-base'
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
# last_hidden_state: torch.FloatTensor = None
# past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
# hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# attentions: Optional[Tuple[torch.FloatTensor]] = None
# cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
caption='a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a '
caption='a a a a a a a a a a a '
input_ids = tokenizer(caption, 
                    #   return_tensors="pt",
                      padding="max_length",
                      truncation=True,
                      max_length=tokenizer.model_max_length).input_ids  # Batch size 1
print(tokenizer.pad_token_id,'pad_token_id')
print(tokenizer.eos_token_id,'eos_token_id')
print(len(input_ids),'input_ids')
print(input_ids,'input_ids')