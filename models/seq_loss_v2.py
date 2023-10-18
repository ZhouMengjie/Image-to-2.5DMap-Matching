import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.utils.module_with_records import ModuleWithRecords

class MultiInfoNCELoss:
    def __init__(self, margin):
        self.img_map_loss = InfoNCELoss(margin)
        self.auxilary_loss = InfoNCELoss_v2(margin)

    def __call__(self, img_feats, map_feats, img_feats_sq, map_feats_sq):
        # Loss on the cross-modal (img-map)
        img_map_loss, img_map_states, _ = self.img_map_loss(img_feats, map_feats)        
        img_map_states = {'img_map_{}'.format(e): img_map_states[e] for e in img_map_states}
        loss = 0.
        stats = img_map_states
        loss += img_map_loss 

        aux_loss, aux_stats, _ = self.auxilary_loss(img_feats_sq, map_feats_sq)
        aux_stats = {'auxilary_{}'.format(e): aux_stats[e] for e in aux_stats}
        stats.update(aux_stats)
        loss += 0.1*aux_loss

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


class InfoNCELoss_v2:
    def __init__(self, margin, normalize_embeddings=True, seq_len=5):
        self.margin = margin 
        self.distance = CustomDistance(normalize_embeddings=normalize_embeddings, collect_stats=True, seq_len=seq_len)
        reducer_fn = reducers.MeanReducer(collect_stats=True)
        self.loss_fn = losses.NTXentLoss(temperature=self.margin, distance=self.distance, reducer=reducer_fn, collect_stats=True)

    def __call__(self, query_emb, ref_emb):
        if ref_emb is None:
            ref_emb = query_emb 
        hard_triplets = None
        query_labels = torch.arange(query_emb.shape[0]).to(query_emb.device)
        ref_labels = torch.arange(ref_emb.shape[0]).to(ref_emb.device)
        query_emb = query_emb.view(query_emb.shape[0], -1)
        ref_emb = ref_emb.view(ref_emb.shape[0], -1)
        loss1 = self.loss_fn(query_emb, query_labels, hard_triplets, ref_emb, ref_labels)
        loss2 = self.loss_fn(ref_emb, ref_labels, hard_triplets, query_emb, query_labels)
        loss = (loss1 + loss2) / 2
        stats = {'loss': loss.item(), # loss is a tensor, loss.item() is a number
                 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_triplets': query_emb.shape[0]*(query_emb.shape[0]-1),
                 }

        return loss, stats, None


class CustomDistance(ModuleWithRecords):
    def __init__(
        self, normalize_embeddings=True, p=2, power=1, is_inverted=False, seq_len=5, **kwargs
    ):
        super().__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self.seq_len = seq_len
        self.add_to_recordable_attributes(list_of_names=["p", "power"], is_stat=False)

    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        query_emb = query_emb.view(query_emb.shape[0]*self.seq_len,-1)
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb = ref_emb.view(ref_emb.shape[0]*self.seq_len,-1)
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(
            query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
        )
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat**self.power
        return mat

    def compute_mat(self, query_emb, ref_emb):
        query_emb = query_emb.view(-1, self.seq_len, query_emb.shape[1])
        ref_emb = ref_emb.view(-1, self.seq_len, ref_emb.shape[1])
        mat = torch.zeros(query_emb.shape[0], ref_emb.shape[0]).to(query_emb.device)
        for i in range(query_emb.shape[0]):
            for j in range(ref_emb.shape[0]):
                q_i = query_emb[i] # [5, 512]
                r_j = ref_emb[j] # [5, 512]
                res = 0.0
                for t in range(q_i.shape[0]):
                    cosine_sim = F.cosine_similarity(q_i[t], r_j[t], dim=0)
                    res = res + cosine_sim.item()
                res = res / q_i.shape[0] 
                mat[i][j] = res
        return mat

    def normalize(self, embeddings, dim=1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)

    def maybe_normalize(self, embeddings, dim=1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings

    def get_norm(self, embeddings, dim=1, **kwargs):
        return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)

    def set_default_stats(
        self, query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
    ):
        if self.collect_stats:
            with torch.no_grad():
                stats_dict = {
                    "initial_avg_query_norm": torch.mean(
                        self.get_norm(query_emb)
                    ).item(),
                    "initial_avg_ref_norm": torch.mean(self.get_norm(ref_emb)).item(),
                    "final_avg_query_norm": torch.mean(
                        self.get_norm(query_emb_normalized)
                    ).item(),
                    "final_avg_ref_norm": torch.mean(
                        self.get_norm(ref_emb_normalized)
                    ).item(),
                }
                self.set_stats(stats_dict)

    def set_stats(self, stats_dict):
        for k, v in stats_dict.items():
            self.add_to_recordable_attributes(name=k, is_stat=True)
            setattr(self, k, v)


