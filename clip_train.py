import torch
import clip_image_loader
import clip_text_model
import clip_image_model
from tokenizers import Tokenizer
from tokenizers.models import BPE

from torch.utils.data import DataLoader

tokenizer = Tokenizer(BPE())
tokenizer = Tokenizer.from_file('clip_tokenizer.json')

image_dataset = clip_image_loader.ImageTextDataset(data_file='Train_GCC-training.tsv')
text_model = clip_text_model.TransformerEncoder(d_model = 256, d_ff = 1024, n_heads = 8, n_layers = 6).cuda()
image_model = clip_image_model.ResNet(clip_image_model.ResidualBlock, [3, 4, 6, 3]).cuda()

n_epochs = 10
batch_size = 16

train_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(text_model.parameters()) + list(image_model.parameters()), lr=1e-4)

for epoch in range(n_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        image, text = data
        text_embeddings = torch.zeros((batch_size, 30), dtype=torch.int32)

        for j in range(len(text)):
            tokens = tokenizer.encode(text[j]).ids
            tokens = torch.tensor(tokens)
            text_embeddings[j][:min(tokens.size()[0], 30)] = tokens[:30]

        latent = text_model(text_embeddings.cuda())
        mean_embedding = latent.mean(dim=-2)
        image_embedding = image_model(image.cuda())

        ground_truth = torch.diag(torch.ones(batch_size))
    
        #shapes are (batch_size, 256) and (batch_size, 256)
        output_preds = image_embedding @ mean_embedding.T
        #output preds is (image, text)
        labels = torch.arange(batch_size).cuda()
        loss_image = loss_fn(output_preds, labels)
        loss_text = loss_fn(output_preds.T, labels)
        loss = (loss_image + loss_text)/2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        print(f"Epoch {epoch}, Iteration {i}, Loss: {running_loss}")
        running_loss = 0.0

        if i%100 == 99:
            torch.save(text_model.state_dict(), 'clip_text_model.pth')
            torch.save(image_model.state_dict(), 'clip_image_model.pth')


        
        

