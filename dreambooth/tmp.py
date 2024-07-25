import torch

# Create a small label tensor for easy validation
inputs=torch.ones((10,2,768))
print(torch.sum(inputs).item())

inputs[0][0]=-100
print(torch.sum(inputs).item())


# labels=torch.zeros((10,2))
# labels[:,0]=1
# labels=labels.bool() #1
# print(torch.sum(inputs[labels]))



# inputs=inputs.view(10,12,2,64)
# inputs=inputs.reshape(120,2,64)


# incorrect=labels.unsqueeze(1) #10,1,2
# # print(incorrect.shape)
# incorrect=incorrect.repeat(1,12,1) #10,12,1
# incorrect=incorrect.reshape(120,2)
# print(torch.sum(inputs[incorrect]),'incorrect')

# correct=labels.unsqueeze(-1) #10,1,2
# # print(correct.shape)
# correct=correct.repeat(1,1,12) #10,12,1
# correct=correct.reshape(120,2)
# print(torch.sum(inputs[correct]),'correct')




# # labels[0]=0
# # labels[1]=1
# # print("Original Labels:")
# # print(labels,'labels')

# # # Expand and repeat the labels
# # labels = labels.unsqueeze(2)  # Add the extra dimension
# # labels_expanded = labels.repeat(1, 1, 3)  # Repeat in the last new dimension
# # labels_expanded = labels_expanded.unsqueeze(1)  # Add dimension for '12'
# # labels_expanded = labels_expanded.repeat(1, 12, 1, 1)  # Repeat for '12' times

# # # Reshape for simplicity in this example (not to 4800, 77, 64, but a smaller shape)
# # labels_final = labels_expanded.reshape(8, 2, 3)
# # print("Expanded Labels:")

# # print(labels_expanded,'labels_expanded')
# # print(labels_expanded.shape,'labels_expanded')