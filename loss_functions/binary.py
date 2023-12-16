import torch

torch.manual_seed(0)

# Binary setting ##############################################################
print(f"{'Setting up binary case':-^80}")
z = torch.randn(5)
yhat = torch.sigmoid(z)
y = torch.Tensor([0, 1, 1, 0, 1])
print(f"{z=}\n{yhat=}\n{y=}\n{'':-^80}")

# First compute the negative log likelihoods using the derived formula
l = -(y * yhat.log() + (1 - y) * (1 - yhat).log())
print(f"{l}")

# Observe that BCELoss and BCEWithLogitsLoss can produce the same results
l_BCELoss_nored = torch.nn.BCELoss(reduction="none")(yhat, y)
l_BCEWithLogitsLoss_nored = torch.nn.BCEWithLogitsLoss(reduction="none")(z, y)
print(f"{l_BCELoss_nored}\n{l_BCEWithLogitsLoss_nored}\n{'':=^80}")