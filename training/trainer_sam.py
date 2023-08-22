import os
import sys
import numpy as np
import torch
import pickle
import pathlib
import torch.distributed as dist
import tempfile
from training.distributed_utils import cleanup, reduce_value
from eval.evaluate_pickle import get_recall

from torch.utils.tensorboard import SummaryWriter

from config.utils import Params, get_datetime
from models.loss import make_loss
from models.sam import SAM
from models.model_factory import model_factory

VERBOSE = False


def tensors_to_numbers(stats, device, distributed=False):   
    if distributed:
        for e in stats:
            if torch.is_tensor(stats[e]):
                stats[e] = reduce_value(stats[e].to(device), average=True)
            else:
                stats[e] = reduce_value(torch.tensor(stats[e], dtype=float).to(device), average=True)
            stats[e] = stats[e].item()
    else:
        stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def do_train(dataloaders, train_sampler, params: Params, use_amp=False):
    # Create model class
    model = model_factory(params)
   
    # Move and initialize the model to the proper device before configuring the optimizer
    device  = params.device
    model.to(device)

    if params.log:
        s = get_datetime()
        if params.load_weights is not None:
            model_name = os.path.split(params.load_weights)[1]
            model_name = model_name.split('.')[0] + '_' + s
        else:
            model_name = 'model_' + s
        weights_path = create_folder('weights')
        model_pathname = os.path.join(weights_path, model_name)
        print('Model device: {}'.format(device))
        print('Model name: {}'.format(model_name))
        if hasattr(model, 'print_info'):
            model.print_info()
        else:
            n_params = sum([param.nelement() for param in model.parameters()])
            print('Number of model parameters: {}'.format(n_params))

        # Initialize TensorBoard writer
        log_path = create_folder('tf_logs')
        logdir = os.path.join(log_path, model_name)
        writer = SummaryWriter(logdir)
               
    if params.load_weights is not None:
        assert os.path.exists(params.load_weights), 'Cannot open network weights: {}'.format(params.load_weights)
        checkpoint = torch.load(params.load_weights, map_location=device)     
        state_dict = {}
        try:
            ckp = checkpoint['model']
            op_ckp = checkpoint['optimizer']
        except:
            ckp = checkpoint
            op_ckp = None
        for k, v in ckp.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            if new_k == 'cloud_fe.backbone.conv0.kernel' and v.shape[1] != params.feat_size:
                new_k = 'ignore' 
            state_dict[new_k] = v
        model.load_state_dict(state_dict, strict=False) 
        print('load pretrained {} model!'.format(params.load_weights))
    else:
        if params.distributed:
            checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
            # if pre-trained weight don't exist, save the weights from rank 0，then load by other ranks to keep the weight initialization consistent
            if params.log:
                torch.save(model.state_dict(), checkpoint_path)
            dist.barrier()
            # Note: map_location should be set, otherwise the first GPU would occupy more resources
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print('load initial weights!')

    # where freeze some layers
    if params.freeze:
        for name, para in model.named_parameters():
            # freeze image_fe
            if 'cloud_fe' not in name:
                para.requires_grad_(False)

    # syncronize the BN
    if params.syncBN:
        # training would be more time-consuming
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    if params.distributed:
        params.lr *= params.world_size

    loss_fn = make_loss(params)

    # convert to DDP model
    if params.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.gpu])

    params_l = []
    params_l.append({'params': model.parameters(), 'initial_lr':params.lr, 'lr': params.lr})

    # Training elements
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if params.optimizer == 'Adam':
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.Adam(params_l)
        else:
            optimizer = torch.optim.Adam(params_l, weight_decay=params.weight_decay)
    elif params.optimizer == 'SGD':
        # SGD with momentum (default momentum = 0.9)
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.SGD(params_l)
        else:
            optimizer = torch.optim.SGD(params_l, weight_decay=params.weight_decay)
    elif params.optimizer == 'AdamW':
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.AdamW(params_l)
        else:
            optimizer = torch.optim.AdamW(params_l, weight_decay=params.weight_decay, amsgrad=False)
    elif params.optimizer == 'SAM':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(params_l, base_optimizer, scaler, weight_decay=params.weight_decay, amsgrad=False, rho=2.5, adaptive=True)
    else:
        raise NotImplementedError('Unsupported optimizer: {}'.format(params.optimizer))
    
    if params.epoch > 1 and op_ckp is not None:
        optimizer.load_state_dict(op_ckp)
        print('load pretrained {} optimizer!'.format(params.load_weights))

    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.t_max+1,
                                                                   eta_min=params.min_lr, last_epoch=params.epoch)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        elif params.scheduler == 'LambdaLR':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + params.epoch -(params.epochs-params.scheduler_milestones)) / float(params.scheduler_milestones + 1)
                return lr_l
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif params.scheduler == 'ExpLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, params.exp_gamma, last_epoch=params.epoch)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    is_validation_set = 'val' in dataloaders
    # Training statistics
    stats = {'train': [], 'val': []}
    best_top1 = 0

    for epoch in range(params.epoch, params.epochs + 1):
        if params.distributed:
            train_sampler.set_epoch(epoch)

        running_stats = train_one_epoch(model, dataloaders['train'], device, optimizer, loss_fn, params, epoch, scaler, use_amp)
        # ******* TRAIN END *******
        # Compute mean stats for the phase
        epoch_stats = {}
        for key in running_stats[0].keys():
            temp = [e[key] for e in running_stats]
            epoch_stats[key] = np.mean(temp)

        stats['train'].append(epoch_stats) 
        if params.log:
            print('Epoch[{0}]/Phase[{1}]'.format(epoch, 'train'))
            # print epoch states for each phase
            for e in epoch_stats:
                s = 'mean_' + e +': {}'
                print(s.format(epoch_stats[e]))

        if is_validation_set:
            val_stats = validate(model, dataloaders['val'], torch.device('cuda:0'), params, epoch)
            # ******* VALIDATE END *******
            # Compute mean stats for the phase
            stats['val'].append(val_stats)
            if params.log:
                print('Epoch[{0}]/Phase[{1}]\t'
                    'top 1 recall: {recall:.4f}\t'
                    'top 1p recall: {one_percent_recall:.4f}\t'
                    'similarity: {similarity:.4f}'.format(
                    epoch, 'val', recall=val_stats['recall'][0], 
                    one_percent_recall=val_stats['one_percent_recall'], similarity=val_stats['average_similarity']))
            
        # ******* EPOCH END *******
        if params.log:
            # save the latest checkpoint
            epoch_model_path = model_pathname + '_latest.pth'
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, epoch_model_path)

            # save the best model
            if val_stats['recall'][0] > best_top1:
                best_top1 = val_stats['recall'][0]
                epoch_model_path = model_pathname + '_best_top1.pth'
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch, 'best_top1':best_top1}
                torch.save(state, epoch_model_path)

            loss_metrics = {'train': stats['train'][-1]['loss']}
            writer.add_scalars('Loss', loss_metrics, epoch)
            writer.add_scalar('Lr', optimizer.param_groups[0]["lr"], epoch)
            if is_validation_set:
                recall_metrics = {'val': stats['val'][-1]['recall'][0]}
                writer.add_scalars('Recall', recall_metrics, epoch)
        
        if scheduler is not None:
            scheduler.step()

    stats = {'train_stats': stats, 'params': params}
    if params.log:
        pickle_path = model_pathname + '_stats.pickle'
        pickle.dump(stats, open(pickle_path, "wb"))

        if params.distributed is True and params.load_weights is None:
            os.remove(checkpoint_path)

    cleanup()

def train_one_epoch(model, dataloader, device, optimizer, loss_fn, params: Params, epoch, scaler, use_amp):
    model.train()
    count_batches = 0
    running_stats = []  # running stats for the current epoch
    for batch in dataloader:
        # batch is (batch_size, n_points, 3) tensor
        # labels is list with indexes of elements forming a batch
        count_batches += 1
        batch_stats = {}

        batch = {e: batch[e].to(device) for e in batch}
        labels = batch['labels']
        positives_mask = batch['positives_mask']
        negatives_mask = batch['negatives_mask']

        optimizer.zero_grad()
        # Compute embeddings of all elements
        with torch.cuda.amp.autocast(enabled=use_amp):
            embeddings = model(batch)
            loss, temp_stats, _ = loss_fn(embeddings, positives_mask, negatives_mask, labels)  

        scaler.scale(loss).backward()           
        # loss.backward()
        if params.optimizer != 'SAM':
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            temp_stats = tensors_to_numbers(temp_stats, device, params.distributed)
            batch_stats.update(temp_stats)
        else:
            optimizer.first_step(zero_grad=True)
            with torch.cuda.amp.autocast(enabled=use_amp):   
                embeddings = model(batch)
                loss, temp_stats, _ = loss_fn(embeddings, positives_mask, negatives_mask, labels) 
            # loss.backward()
            scaler.scale(loss).backward()  
            optimizer.second_step(zero_grad=True)
            scaler.update()
            temp_stats = tensors_to_numbers(temp_stats, device, params.distributed)
            batch_stats.update(temp_stats)
                        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        
        if params.log:
            print('Epoch[{0}-{1}]\t'
                'loss 1: {img_map_loss:.4f}\t'
                'loss 2: {map_loss:.4f}\t'
                'loss 3: {image_loss:.4f}\t'
                'total loss: {total_loss:.4f}'.format( # check shuffle
                epoch, count_batches, img_map_loss=batch_stats['img_map_loss'],
                map_loss=batch_stats['map_loss'], image_loss=batch_stats['image_loss'], 
                total_loss=batch_stats['loss']))

        running_stats.append(batch_stats)
        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
    # Wait for all processes to finish calculation
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
    return running_stats 


def validate(model, dataloader, device, params:Params, epoch):
    model.eval()
    database_embeddings = []
    query_embeddings = []
    count_batches = 0    
    for batch in dataloader:       
        count_batches += 1
        with torch.no_grad():
            batch = {e: batch[e].to(device) for e in batch}
            x, _, _ = model(batch)
            cloud_embedding = x['embedding']
            image_embedding = x['image_embedding']
            if params.normalize_embeddings:
                cloud_embedding = torch.nn.functional.normalize(cloud_embedding, p=2, dim=1)  # Normalize embeddings
                image_embedding = torch.nn.functional.normalize(image_embedding, p=2, dim=1)

            cloud_embedding = cloud_embedding.detach().cpu().numpy()
            image_embedding = image_embedding.detach().cpu().numpy()

        torch.cuda.empty_cache()  
        database_embeddings.append(cloud_embedding)
        query_embeddings.append(image_embedding)
        if params.log:
            print('Epoch[{0}]/Batch[{1}]'.format(epoch, count_batches))

    database_embeddings = np.vstack(database_embeddings)
    query_embeddings = np.vstack(query_embeddings)

    recall, similarity, one_percent_recall = get_recall(database_embeddings, query_embeddings)
    average_similarity = np.mean(similarity)
    running_stats = {'one_percent_recall': one_percent_recall, 'average_similarity': average_similarity,
                     'recall': recall}  

    # Wait for all processes to finish calculation
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    return running_stats


def create_folder(folder_name):
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    folder_path = os.path.join(temp, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    assert os.path.exists(folder_path), 'Cannot create folder: {}'.format(folder_path)
    return folder_path