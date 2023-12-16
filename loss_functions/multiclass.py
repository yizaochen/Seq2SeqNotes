import torch


# Multiclass setting ##########################################################
print(f"{'Setting up multiclass case':-^80}")
z2 = torch.randn(5, 3)
yhat2 = torch.softmax(z2, dim=-1)
y2 = torch.Tensor([0, 2, 1, 1, 0]).long()
print(f"{z2=}\n{yhat2=}\n{y2=}\n{'':-^80}")

# First compute the negative log likelihoods using the derived formulat
l2 = -yhat2.log()[torch.arange(5), y2]  # masking the correct entries
print(f"{l2}")
print(-torch.log_softmax(z2, dim=-1)[torch.arange(5), y2])

# Observe that NLLLoss and CrossEntropyLoss can produce the same results
l2_NLLLoss_nored = torch.nn.NLLLoss(reduction="none")(yhat2.log(), y2)
l2_CrossEntropyLoss_nored = torch.nn.CrossEntropyLoss(reduction="none")(z2, y2)
print(f"{l2_NLLLoss_nored}\n{l2_CrossEntropyLoss_nored}\n{'':=^80}")