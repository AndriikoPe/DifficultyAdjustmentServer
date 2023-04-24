import torch

model = torch.load('trained_model_1.pt')

input_data = torch.tensor([0.23392242677397193,
                           0.0017457588789563192,
                           133.9946939945221 / 15116.16,
                           0.10708058379661198,
                           0.08511973035844755,
                           -0.3385072974576493,
                           1.3684985439838737 / 2.0]).reshape(1, 7)

output1, output2 = model(input_data)
action = output2.detach().numpy()[0][0]
print(action)
