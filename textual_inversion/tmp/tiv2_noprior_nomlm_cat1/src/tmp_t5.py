from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
# last_hidden_state: torch.FloatTensor = None
# past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
# hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# attentions: Optional[Tuple[torch.FloatTensor]] = None
# cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
caption='summarize: studies have shown that owning a dog is good for you'
input_ids = tokenizer(caption, 
                      return_tensors="pt",
                      padding="max_length",
                      max_length=tokenizer.model_max_length).input_ids  # Batch size 1
print(input_ids.shape ,'input_ids ')
print(tokenizer.model_max_length ,'model_max_length ')
# the forward function automatically creates the correct decoder_input_ids
outputs = model.encoder(input_ids,output_hidden_states=True)
print(outputs.last_hidden_state.shape,'outputs.last_hidden_state.shape')
for item in outputs.hidden_states:
    print(item.shape,'hidden_states')
print(input_ids.shape,'input_idss.hape',tokenizer.model_max_length)