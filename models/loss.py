# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import LpDistance, DotProductSimilarity


def make_loss(params):
    if params.loss == 'MultiBatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = MultiBatchHardTripletLoss(params.margin, params.normalize_embeddings, params.weights)
    elif params.loss == 'MultiInfoNCELoss':
        loss_fn = MultiInfoNCELoss(params.margin, params.normalize_embeddings, params.weights)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError
    return loss_fn


class MultiBatchHardTripletLoss:
    def __init__(self, margin, normalize_embeddings, weights):
        assert len(weights) == 4
        self.weights = weights
        self.img_cloud_loss = BatchHardTripletLoss(margin, normalize_embeddings)
        self.cloud_img_loss = BatchHardTripletLoss(margin, normalize_embeddings)  
        self.cloud_loss = BatchHardTripletLoss(margin, normalize_embeddings)
        self.image_loss = BatchHardTripletLoss(margin, normalize_embeddings)

    def __call__(self, x, positives_mask, negatives_mask, labels):
        # Loss on the image-cloud
        img_cloud_loss, img_cloud_states, _ = self.img_cloud_loss(x['image_embedding'], x['cloud_embedding'], positives_mask, negatives_mask, labels)        
        img_cloud_states = {'img_cloud_{}'.format(e): img_cloud_states[e] for e in img_cloud_states}
        loss = 0.
        stats = img_cloud_states
        if self.weights[0] > 0.:
            loss = self.weights[0] * img_cloud_loss + loss

        # Loss on the cloud-image
        cloud_img_loss, cloud_img_states, _ = self.cloud_img_loss(x['cloud_embedding'], x['image_embedding'], positives_mask, negatives_mask, labels)        
        cloud_img_states = {'cloud_img_{}'.format(e): cloud_img_states[e] for e in cloud_img_states}
        stats.update(cloud_img_states)
        if self.weights[1] > 0.:
            loss = self.weights[1] * cloud_img_loss + loss        

        # Loss on the cloud-based descriptor
        cloud_loss, cloud_stats, _ = self.cloud_loss(x['cloud_embedding'], None, positives_mask, negatives_mask, labels)
        cloud_stats = {'cloud_{}'.format(e): cloud_stats[e] for e in cloud_stats}
        stats.update(cloud_stats)
        if self.weights[2] > 0.:
            loss = self.weights[2] * cloud_loss + loss

        # Loss on the image-based descriptor
        image_loss, image_stats, _ = self.image_loss(x['image_embedding'], None, positives_mask, negatives_mask, labels)
        image_stats = {'image_{}'.format(e): image_stats[e] for e in image_stats}
        stats.update(image_stats)
        if self.weights[3] > 0.:
            loss = self.weights[3] * image_loss + loss

        stats['loss'] = loss.item()
        return loss, stats, None


class BatchHardTripletLoss:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin # we set it to 1
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings, collect_stats=True)
        # We use triplet loss with Euclidean distance
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, smooth_loss=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, query_emb, ref_emb, positives_mask, negatives_mask, labels):
        if ref_emb is None:
            ref_emb = query_emb 
        hard_triplets = None
        query_labels = torch.arange(query_emb.shape[0]).to(query_emb.device)
        ref_labels = torch.arange(ref_emb.shape[0]).to(ref_emb.device) 
        loss = self.loss_fn(query_emb, query_labels, hard_triplets, ref_emb, ref_labels)
        stats = {'loss': loss.item(), # loss is a tensor, loss.item() is a number
                 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': query_emb.shape[0]*(query_emb.shape[0]-1),
                 'normalized_loss': loss.item() * self.loss_fn.reducer.triplets_past_filter,
                 'total_loss': self.loss_fn.reducer.loss * self.loss_fn.reducer.triplets_past_filter
                 }

        return loss, stats, hard_triplets


class MultiInfoNCELoss:
    def __init__(self, margin, normalize_embeddings, weights):
        assert len(weights) == 4
        self.weights = weights
        self.img_cloud_loss = InfoNCELoss(margin, normalize_embeddings)
        self.cloud_img_loss = InfoNCELoss(margin, normalize_embeddings)  
        self.cloud_loss = InfoNCELoss(margin, normalize_embeddings)
        self.image_loss = InfoNCELoss(margin, normalize_embeddings)

    def __call__(self, x, positives_mask, negatives_mask, labels):
         # Loss on the image-cloud
        img_cloud_loss, img_cloud_states, _ = self.img_cloud_loss(x['image_embedding'], x['cloud_embedding'], positives_mask, negatives_mask, labels)        
        img_cloud_states = {'img_cloud_{}'.format(e): img_cloud_states[e] for e in img_cloud_states}
        loss = 0.
        stats = img_cloud_states
        if self.weights[0] > 0.:
            loss = self.weights[0] * img_cloud_loss + loss

        # Loss on the cloud-image
        cloud_img_loss, cloud_img_states, _ = self.cloud_img_loss(x['cloud_embedding'], x['image_embedding'], positives_mask, negatives_mask, labels)        
        cloud_img_states = {'cloud_img_{}'.format(e): cloud_img_states[e] for e in cloud_img_states}
        stats.update(cloud_img_states)
        if self.weights[1] > 0.:
            loss = self.weights[1] * cloud_img_loss + loss        

        # Loss on the cloud-based descriptor
        cloud_loss, cloud_stats, _ = self.cloud_loss(x['cloud_embedding'], None, positives_mask, negatives_mask, labels)
        cloud_stats = {'cloud_{}'.format(e): cloud_stats[e] for e in cloud_stats}
        stats.update(cloud_stats)
        if self.weights[2] > 0.:
            loss = self.weights[2] * cloud_loss + loss

        # Loss on the image-based descriptor
        image_loss, image_stats, _ = self.image_loss(x['image_embedding'], None, positives_mask, negatives_mask, labels)
        image_stats = {'image_{}'.format(e): image_stats[e] for e in image_stats}
        stats.update(image_stats)
        if self.weights[3] > 0.:
            loss = self.weights[3] * image_loss + loss

        stats['loss'] = loss.item()
        return loss, stats, None


class InfoNCELoss:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin # we set it to 0.1
        self.distance = DotProductSimilarity(normalize_embeddings=normalize_embeddings, collect_stats=True)
        # We use triplet loss with Euclidean distance
        reducer_fn = reducers.MeanReducer(collect_stats=True)
        self.loss_fn = losses.NTXentLoss(temperature=self.margin, distance=self.distance, reducer=reducer_fn, collect_stats=True)

    def __call__(self, query_emb, ref_emb, positives_mask, negatives_mask, labels):
        if ref_emb is None:
            ref_emb = query_emb 
        hard_triplets = None
        query_labels = torch.arange(query_emb.shape[0]).to(query_emb.device)
        ref_labels = torch.arange(ref_emb.shape[0]).to(ref_emb.device) 
        loss = self.loss_fn(query_emb, query_labels, hard_triplets, ref_emb, ref_labels)
        stats = {'loss': loss.item(), # loss is a tensor, loss.item() is a number
                 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_triplets': query_emb.shape[0]*(query_emb.shape[0]-1),
                 }

        return loss, stats, hard_triplets
