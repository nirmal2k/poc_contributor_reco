import pickle
from network_train import network_embedding
import utils
import math
import random
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer

def main():
    device='cuda'

    all_uids, user_emb, all_pids, post_embs, user_user_samples, user_post_samples = pickle.load(open('./so_tr_data.pkl', 'rb'))

    
    uu_bs = 64
    num_epochs = 25
    up_bs = math.ceil(len(user_post_samples)/(len(user_user_samples)/uu_bs))

    model = network_embedding(user_emb, post_embs, device)
    optimizer, scheduler = utils.get_optimizer_scheduler([model], len(user_user_samples), 0.025, uu_bs, num_epochs)

    writer = SummaryWriter('./Models')
    for e in range(num_epochs):        
        random.shuffle(user_user_samples)
        random.shuffle(user_post_samples)

        buser_user_samples = list(utils.chunks(user_user_samples, uu_bs))        
        buser_post_samples = list(utils.chunks(user_post_samples, up_bs))

        total_steps = min(len(buser_user_samples), len(buser_post_samples))
        print(total_steps)

        for batch_num, [uu, up] in enumerate(tqdm(list(zip(buser_user_samples, buser_post_samples)))):
            optimizer.zero_grad()

            uu_loss = model.forward(uu,'user')
            up_loss = model.forward(up,'post')
            loss = uu_loss+up_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_num % 10 == 0:  
                writer.add_scalar('User user', uu_loss.item(), (e)*total_steps+batch_num)
                writer.add_scalar('Post post', up_loss.item(), (e)*total_steps+batch_num)             
                writer.add_scalar('loss', loss.item(), (e)*total_steps+batch_num)

        
        torch.save(model.state_dict(), './Models/model.pt')

if __name__ == '__main__':
    main()