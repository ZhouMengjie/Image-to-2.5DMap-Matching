import torch
import torch.nn as nn
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import LpDistance, DotProductSimilarity
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
CE = torch.nn.BCELoss(reduction='sum')


def make_loss(params):
    if params.loss == 'MultiBatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = MultiBatchHardTripletLoss(params.margin, params.normalize_embeddings, params.weights, params.batch_size)
    elif params.loss == 'MultiInfoNCELoss':
        loss_fn = MultiInfoNCELoss(params.margin, params.normalize_embeddings, params.weights, params.batch_size)
    elif params.loss == 'MultiMIRInfoNCELoss':
        loss_fn = MultiMIRInfoNCELoss(params.margin, params.normalize_embeddings, params.weights, params.batch_size, params.feat_dim)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError
    return loss_fn


class MultiBatchHardTripletLoss:
    def __init__(self, margin, normalize_embeddings, weights, batch_size):
        assert len(weights) == 4
        self.batch_size = batch_size
        self.weights = weights
        self.img_map_loss = BatchHardTripletLoss(margin, normalize_embeddings)
        self.map_loss = BatchHardTripletLoss(margin, normalize_embeddings)
        self.image_loss = BatchHardTripletLoss(margin, normalize_embeddings)

    def __call__(self, x, positives_mask, negatives_mask, labels):
        map_t1_feats = x['embedding'][:self.batch_size, :]
        map_t2_feats = x['embedding'][self.batch_size: , :]
        map_feats = torch.stack([map_t1_feats,map_t2_feats]).mean(dim=0)
        
        img_t1_feats = x['image_embedding'][:self.batch_size, :]
        img_t2_feats = x['image_embedding'][self.batch_size: , :]
        img_feats = torch.stack([img_t1_feats,img_t2_feats]).mean(dim=0)

        # Loss on the cross-modal (img-map)
        img_map_loss, img_map_states, _ = self.img_map_loss(map_feats, img_feats, positives_mask, negatives_mask, labels)        
        img_map_states = {'img_map_{}'.format(e): img_map_states[e] for e in img_map_states}
        loss = 0.
        stats = img_map_states
        if self.weights[0] > 0.:
            loss = self.weights[0] * img_map_loss + loss

        # Loss on the intra-modal (map)
        map_loss, map_stats, _ = self.map_loss(map_t1_feats, map_t2_feats, positives_mask, negatives_mask, labels)
        map_stats = {'map_{}'.format(e): map_stats[e] for e in map_stats}
        stats.update(map_stats)
        if self.weights[1] > 0.:
            loss = self.weights[1] * map_loss + loss

        # Loss on the intra-modal (image)
        image_loss, image_stats, _ = self.image_loss(img_t1_feats, img_t2_feats, positives_mask, negatives_mask, labels)
        image_stats = {'image_{}'.format(e): image_stats[e] for e in image_stats}
        stats.update(image_stats)
        if self.weights[2] > 0.:
            loss = self.weights[2] * image_loss + loss

        stats['loss'] = loss.item()
        return loss, stats, None


class BatchHardTripletLoss:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin # we set it to 1
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings, collect_stats=True)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, smooth_loss=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, query_emb, ref_emb, positives_mask, negatives_mask, labels):
        if ref_emb is None:
            ref_emb = query_emb 
        hard_triplets = None
        # hard_triplets1 = self.miner_fn(query_emb, ref_emb, positives_mask, negatives_mask)
        # hard_triplets2 = self.miner_fn(ref_emb, query_emb, positives_mask, negatives_mask)
        query_labels = torch.arange(query_emb.shape[0]).to(query_emb.device)
        ref_labels = torch.arange(ref_emb.shape[0]).to(ref_emb.device) 
        loss1 = self.loss_fn(query_emb, query_labels, hard_triplets, ref_emb, ref_labels)
        loss2 = self.loss_fn(ref_emb, ref_labels, hard_triplets, query_emb, query_labels)
        loss = (loss1 + loss2) / 2

        stats = {'loss': loss.item(), # loss is a tensor, loss.item() is a number
                 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': query_emb.shape[0]*(query_emb.shape[0]-1),
                #  'num_triplets': len(hard_triplets1[0]) + len(hard_triplets2[0]) / 2,
                 'normalized_loss': loss.item() * self.loss_fn.reducer.triplets_past_filter,
                 'total_loss': self.loss_fn.reducer.loss * self.loss_fn.reducer.triplets_past_filter
                 }

        return loss, stats, None


class MultiInfoNCELoss:
    def __init__(self, margin, normalize_embeddings, weights, batch_size):
        assert len(weights) == 3 # for pickle case, len(weights) == 3
        self.batch_size = batch_size
        self.weights = weights 
        self.img_map_loss = InfoNCELoss(margin, normalize_embeddings)
        self.map_loss = InfoNCELoss(margin, normalize_embeddings)
        self.image_loss = InfoNCELoss(margin, normalize_embeddings)

    def __call__(self, x, positives_mask, negatives_mask, labels):
        # Splite embeddings
        map_t1_feats = x['embedding'][:self.batch_size, :]
        map_t2_feats = x['embedding'][self.batch_size: , :]
        map_feats = torch.stack([map_t1_feats,map_t2_feats]).mean(dim=0)
        
        img_t1_feats = x['image_embedding'][:self.batch_size, :]
        img_t2_feats = x['image_embedding'][self.batch_size: , :]
        img_feats = torch.stack([img_t1_feats,img_t2_feats]).mean(dim=0)

        # Loss on the cross-modal (img-map)
        img_map_loss, img_map_states, _ = self.img_map_loss(map_feats, img_feats, positives_mask, negatives_mask, labels)        
        img_map_states = {'img_map_{}'.format(e): img_map_states[e] for e in img_map_states}
        loss = 0.
        stats = img_map_states
        if self.weights[0] > 0.:
            loss = self.weights[0] * img_map_loss + loss

        # Loss on the intra-modal (map)
        map_loss, map_stats, _ = self.map_loss(map_t1_feats, map_t2_feats, positives_mask, negatives_mask, labels)
        map_stats = {'map_{}'.format(e): map_stats[e] for e in map_stats}
        stats.update(map_stats)
        if self.weights[1] > 0.:
            loss = self.weights[1] * map_loss + loss

        # Loss on the intra-modal (image)
        image_loss, image_stats, _ = self.image_loss(img_t1_feats, img_t2_feats, positives_mask, negatives_mask, labels)
        image_stats = {'image_{}'.format(e): image_stats[e] for e in image_stats}
        stats.update(image_stats)
        if self.weights[2] > 0.:
            loss = self.weights[2] * image_loss + loss

        stats['loss'] = loss.item()
        return loss, stats, None


class MultiMIRInfoNCELoss:
    def __init__(self, margin, normalize_embeddings, weights, batch_size, input_channels):
        assert len(weights) == 4 # for pickle case, len(weights) == 3
        self.batch_size = batch_size
        self.weights = weights 
        self.img_map_loss = InfoNCELoss(margin, normalize_embeddings)
        self.map_loss = InfoNCELoss(margin, normalize_embeddings)
        self.image_loss = InfoNCELoss(margin, normalize_embeddings)
        self.latent_loss = MutualInfoReg(input_channels)

    def __call__(self, x, positives_mask, negatives_mask, labels):
        # Splite embeddings
        map_t1_feats = x['embedding'][:self.batch_size, :]
        map_t2_feats = x['embedding'][self.batch_size: , :]
        map_feats = torch.stack([map_t1_feats,map_t2_feats]).mean(dim=0)
        
        img_t1_feats = x['image_embedding'][:self.batch_size, :]
        img_t2_feats = x['image_embedding'][self.batch_size: , :]
        img_feats = torch.stack([img_t1_feats,img_t2_feats]).mean(dim=0)

        tile_t1_feats = x['tile_embedding'][:self.batch_size, :]
        tile_t2_feats = x['tile_embedding'][self.batch_size: , :]
        tile_feats = torch.stack([tile_t1_feats,tile_t2_feats]).mean(dim=0)

        pc_t1_feats = x['cloud_embedding'][:self.batch_size, :]
        pc_t2_feats = x['cloud_embedding'][self.batch_size: , :]
        pc_feats = torch.stack([pc_t1_feats,pc_t2_feats]).mean(dim=0)

        # Loss on the cross-modal (img-map)
        img_map_loss, img_map_states, _ = self.img_map_loss(map_feats, img_feats, positives_mask, negatives_mask, labels)        
        img_map_states = {'img_map_{}'.format(e): img_map_states[e] for e in img_map_states}
        loss = 0.
        stats = img_map_states
        if self.weights[0] > 0.:
            loss = self.weights[0] * img_map_loss + loss

        # Loss on the intra-modal (map)
        map_loss, map_stats, _ = self.map_loss(map_t1_feats, map_t2_feats, positives_mask, negatives_mask, labels)
        map_stats = {'map_{}'.format(e): map_stats[e] for e in map_stats}
        stats.update(map_stats)
        if self.weights[1] > 0.:
            loss = self.weights[1] * map_loss + loss

        # Loss on the intra-modal (image)
        image_loss, image_stats, _ = self.image_loss(img_t1_feats, img_t2_feats, positives_mask, negatives_mask, labels)
        image_stats = {'image_{}'.format(e): image_stats[e] for e in image_stats}
        stats.update(image_stats)
        if self.weights[2] > 0.:
            loss = self.weights[2] * image_loss + loss

        # Loss on the latent space (latent)
        latent_loss, latent_stats, _ = self.latent_loss(tile_feats, pc_feats)
        latent_stats = {'latent_loss': latent_loss.item()}
        stats.update(latent_stats)
        if self.weights[3] > 0.:
            loss = self.weights[3] * latent_loss + loss

        stats['loss'] = loss.item()
        return loss, stats, None


class InfoNCELoss:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin # we set it to 0.1
        self.distance = DotProductSimilarity(normalize_embeddings=normalize_embeddings, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        # We use triplet loss with Euclidean distance
        reducer_fn = reducers.MeanReducer(collect_stats=True)
        self.loss_fn = losses.NTXentLoss(temperature=self.margin, distance=self.distance, reducer=reducer_fn, collect_stats=True)

    def __call__(self, query_emb, ref_emb, positives_mask, negatives_mask, labels):
        if ref_emb is None:
            ref_emb = query_emb 
        hard_triplets = None
        # hard_triplets1 = self.miner_fn(query_emb, ref_emb, positives_mask, negatives_mask)
        # hard_triplets2 = self.miner_fn(ref_emb, query_emb, positives_mask, negatives_mask)
        query_labels = torch.arange(query_emb.shape[0]).to(query_emb.device)
        ref_labels = torch.arange(ref_emb.shape[0]).to(ref_emb.device) 
        loss1 = self.loss_fn(query_emb, query_labels, hard_triplets, ref_emb, ref_labels)
        loss2 = self.loss_fn(ref_emb, ref_labels, hard_triplets, query_emb, query_labels)
        loss = (loss1 + loss2) / 2
        stats = {'loss': loss.item(), # loss is a tensor, loss.item() is a number
                 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_triplets': query_emb.shape[0]*(query_emb.shape[0]-1),
                #  'num_triplets': len(hard_triplets1[0]) + len(hard_triplets2[0]) / 2
                 }

        return loss, stats, None


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, query_emb, ref_emb, positives_mask, negatives_mask):
        d_query_emb = query_emb.detach()
        d_ref_emb = ref_emb.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_query_emb, d_ref_emb, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, query_emb, ref_emb, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(query_emb, ref_emb)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class MutualInfoReg(nn.Module):
    def __init__(self, input_channels, latent_size=6):
        super(MutualInfoReg, self).__init__()
        self.fc1_img = nn.Linear(input_channels, latent_size)
        self.fc2_img = nn.Linear(input_channels, latent_size)
        self.fc1_pc = nn.Linear(input_channels, latent_size)
        self.fc2_pc = nn.Linear(input_channels, latent_size)
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, img_feat, pc_feat):
        mu_rgb = self.fc1_img(img_feat)
        logvar_rgb = self.fc2_img(img_feat)
        mu_depth = self.fc1_pc(pc_feat)
        logvar_depth = self.fc2_pc(pc_feat)
        
        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld

        return latent_loss
