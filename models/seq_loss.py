import torch
import torch.nn as nn
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import DotProductSimilarity


class MultiInfoNCELoss:
    def __init__(self, margin):
        self.img_map_loss = InfoNCELoss(margin)

    def __call__(self, img_feats, map_feats):
        # Loss on the cross-modal (img-map)
        img_map_loss, img_map_states, _ = self.img_map_loss(img_feats, map_feats)        
        img_map_states = {'img_map_{}'.format(e): img_map_states[e] for e in img_map_states}
        loss = 0.
        stats = img_map_states
        loss += img_map_loss 
        stats['loss'] = loss.item()
        return loss, stats, None


class InfoNCELoss:
    def __init__(self, margin, normalize_embeddings=True):
        self.margin = margin 
        self.distance = DotProductSimilarity(normalize_embeddings=normalize_embeddings, collect_stats=True)
        reducer_fn = reducers.MeanReducer(collect_stats=True)
        self.loss_fn = losses.NTXentLoss(temperature=self.margin, distance=self.distance, reducer=reducer_fn, collect_stats=True)

    def __call__(self, query_emb, ref_emb):
        if ref_emb is None:
            ref_emb = query_emb 
        hard_triplets = None
        query_labels = torch.arange(query_emb.shape[0]).to(query_emb.device)
        ref_labels = torch.arange(ref_emb.shape[0]).to(ref_emb.device) 
        loss1 = self.loss_fn(query_emb, query_labels, hard_triplets, ref_emb, ref_labels)
        loss2 = self.loss_fn(ref_emb, ref_labels, hard_triplets, query_emb, query_labels)
        loss = (loss1 + loss2) / 2
        stats = {'loss': loss.item(), # loss is a tensor, loss.item() is a number
                 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_triplets': query_emb.shape[0]*(query_emb.shape[0]-1),
                 }

        return loss, stats, None


