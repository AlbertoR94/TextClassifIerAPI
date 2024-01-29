from preprocessing import *
from model import *

data = '{"data":[{"id":0, "text":"Angelina Jolie Ready To Adopt More Kids Once Divorce From Brad Pitt Is Behind Her"}, {"id":1, "text":"Netflix: Movies and TV"}]}'

my_dataset = myDataset(data, text_transform)

print(my_dataset[0])

BATCH_SIZE = len(my_dataset)

dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, collate_fn=applyPadding)

for batch in dataloader:
    print(batch.shape)

print(convClassifier.embedding.weight)

convClassifier.eval()
with torch.inference_mode():
    print(torch.sigmoid(convClassifier(batch.to(device))).tolist())

