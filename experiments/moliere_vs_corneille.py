from rnn_language_model import *
from torch.utils.data import DataLoader, Dataset
import re

if __name__ == "__main__":
    import os
    if not (os.path.isdir('moliere_vs_corneille') and
            os.path.isfile(os.path.join('moliere_vs_corneille', 'backoff.pkl'))):
        if not os.path.isdir('moliere_vs_corneille'): os.mkdir('moliere_vs_corneille')

        moliere_files = [f"molière_{i}.txt" for i in range(1,5)]
        corneille_files = [f"corneille_{i}.txt" for i in range(1,8)]

        def read_files(file_pathes, splitter = r"[\.;!\?~\*]"):
            data = []
            for path in file_pathes:
                with open(path, 'r', encoding="utf8") as f:
                    sentences = [sentence.strip() for sentence in re.split(splitter, f.read()) if sentence]
                    data += sentences
            return data

        print(moliere_files)
        moliere_data = read_files(moliere_files)
        print(len(moliere_data), moliere_data[0])

        print(corneille_files)
        corneille_data = read_files(corneille_files)
        print(len(corneille_data), corneille_data[0])

        encoder = CharEncoder()
        encoder.train(moliere_data + corneille_data)


        # small dataset from a list
        class Data(Dataset):
            def __init__(self, data):
                self.data = data
            def __getitem__(self, index):
                return self.data[index]
            def __len__(self):
                return len(self.data)
        
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        #DEVICE = "cpu"

        n_epoch = 10
        batch_size = 128
        learning_rate = 1e-3
        collate = lambda sample: torch.tensor(
            encoder.add_padding([encoder.encode(sentence) for sentence in sample]),
            dtype=torch.long, device=DEVICE) # encode, pad and transform in a tensor
        
        full_dataset = Data(moliere_data + corneille_data) # larger dataset of the full data
        moliere_dataset = Data(moliere_data) # dataset of the moliere-only data
        corneille_dataset = Data(corneille_data) # dataset of the corneille-only data
        full_dataloader = DataLoader(full_dataset, collate_fn=collate, shuffle=True, batch_size=batch_size)
        moliere_dataloader = DataLoader(moliere_dataset, collate_fn=collate, shuffle=True, batch_size=batch_size)
        corneille_dataloader = DataLoader(corneille_dataset, collate_fn=collate, shuffle=True, batch_size=batch_size)

        full_lm = LanguageModel(len(encoder), 16, 128, 2, 0.2)
        moliere_lm = LanguageModel(len(encoder), 16, 128, 2, 0.2)
        corneille_lm = LanguageModel(len(encoder), 16, 128, 2, 0.2)

        backoff_lm = BackoffMechanism()
        backoff_lm.add(full_lm, "full")
        backoff_lm.add(moliere_lm, "moliere", "full")
        backoff_lm.add(corneille_lm, "corneille", "full")

        # train the two models
        for model_name, model, dataloader in [
                ("fufu", full_lm.to(DEVICE), full_dataloader),
                ("momo", moliere_lm.to(DEVICE), moliere_dataloader),
                ("coco", corneille_lm.to(DEVICE), corneille_dataloader)]:
            print(f"training '{model_name}' model")
            model.train()
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(n_epoch):
                running_loss, samples = 0, 0
                for sample in dataloader:
                    optimizer.zero_grad()
                    output, hidden = model(sample[:,:-1])
                    loss = loss_function(output.transpose(-1, -2), sample[:,1:]) # train to predict next character
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.cpu().item()
                    samples += 1
                print(f"Epoch {epoch+1}/{n_epoch}: loss = {running_loss/samples}")
            model.eval()

        backoff_lm.save('moliere_vs_corneille', 'backoff.pkl')
        encoder.save(os.path.join('moliere_vs_corneille', 'encoder.pkl'))

    else:
        DEVICE = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
        backoff_lm = BackoffMechanism.load('moliere_vs_corneille', 'backoff.pkl', device=DEVICE)
        encoder = CharEncoder.load(os.path.join('moliere_vs_corneille', 'encoder.pkl'))

        backoff_lm.backoff_rate = 0.4

        # generate next word using backoff
        print(backoff_lm.sample("full", encoder, device=DEVICE))
        print(backoff_lm.sample("moliere", encoder, device=DEVICE))
        print(backoff_lm.sample("corneille", encoder, device=DEVICE))
        print("---")
        print("Je", backoff_lm.sample("full", encoder, "Je", device=DEVICE))
        print("Je", backoff_lm.sample("moliere", encoder, "Je", device=DEVICE))
        print("Je", backoff_lm.sample("corneille", encoder, "Je", device=DEVICE))
        print("---")
        ctx = "Je"
        print(ctx, backoff_lm.sample("full", encoder, ctx, 15, device=DEVICE))
        print(ctx, backoff_lm.sample("moliere", encoder, ctx, 10, device=DEVICE))
        print(ctx, backoff_lm.sample("corneille", encoder, ctx, 10, device=DEVICE))
        print("---")
        ctx = "Je vous"
        print(ctx, backoff_lm.sample("full", encoder, ctx, 15, device=DEVICE))
        print(ctx, backoff_lm.sample("moliere", encoder, ctx, 10, device=DEVICE))
        print(ctx, backoff_lm.sample("corneille", encoder, ctx, 10, device=DEVICE))
        print("---")
        ctx = "Je vous parle"
        print(ctx, backoff_lm.sample("full", encoder, ctx, 15, device=DEVICE))
        print(ctx, backoff_lm.sample("moliere", encoder, ctx, 10, device=DEVICE))
        print(ctx, backoff_lm.sample("corneille", encoder, ctx, 10, device=DEVICE))
        print("---")
        ctx = "Je vous parle"
        print(ctx, backoff_lm.sample("full", encoder, ctx, 15, strategy="random", device=DEVICE))
        print(ctx, backoff_lm.sample("moliere", encoder, ctx, 10, strategy="random", device=DEVICE))
        print(ctx, backoff_lm.sample("corneille", encoder, ctx, 10, strategy="random", device=DEVICE))
        print("---")
        ctx = "Je vous dis"
        print(ctx, backoff_lm.sample("full", encoder, ctx, 15, strategy="random", device=DEVICE))
        print(ctx, backoff_lm.sample("moliere", encoder, ctx, 10, strategy="random", device=DEVICE))
        print(ctx, backoff_lm.sample("corneille", encoder, ctx, 10, strategy="random", device=DEVICE))