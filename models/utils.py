import torch
from torch.nn.functional import grid_sample
mesh_grid_cache = {}

def batch_indexing(batched_data: torch.Tensor, batched_indices: torch.Tensor, layout='channel_first'):
    def batch_indexing_channel_first(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, C, N]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, C, I1, I2, ..., Im]
        """
        def product(arr):
            p = 1
            for i in arr:
                p *= i
            return p
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size, n_channels = batched_data.shape[:2]
        indices_shape = list(batched_indices.shape[1:])
        batched_indices = batched_indices.reshape([batch_size, 1, -1])
        batched_indices = batched_indices.expand([batch_size, n_channels, product(indices_shape)])
        result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
        result = result.view([batch_size, n_channels] + indices_shape)
        return result

    def batch_indexing_channel_last(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, N, C]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, I1, I2, ..., Im, C]
        """
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size = batched_data.shape[0]
        view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
        expand_shape = [batch_size] + list(batched_indices.shape)[1:]
        indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
        indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)  # [bs, I1, I2, ..., Im]
        if len(batched_data.shape) == 2:
            return batched_data[indices_of_batch, batched_indices.to(torch.long)]
        else:
            return batched_data[indices_of_batch, batched_indices.to(torch.long), :]

    if layout == 'channel_first':
        return batch_indexing_channel_first(batched_data, batched_indices)
    elif layout == 'channel_last':
        return batch_indexing_channel_last(batched_data, batched_indices)
    else:
        raise ValueError


def projection(center, pc, image_h, image_w, sensor_h, sensor_w, npoints, batch_size):
        center_x = center[:,0]
        center_y = center[:,2]
        cx = 0.5*sensor_w - center_x
        cy = 0.5*sensor_h - center_y
        cx = cx.view(batch_size, 1)
        cy = cy.view(batch_size, 1)
        image_x = (pc[:,0,:] + cx)*(image_w - 1) / (sensor_w - 1)
        image_y = (pc[:,2,:] + cy)*(image_h - 1) / (sensor_h - 1)
        return torch.cat([
            image_x[:,None,:],
            image_y[:,None,:],
        ], dim=1)


def mesh_grid(n, h, w, device, channel_first=True):
    global mesh_grid_cache
    str_id = '%d,%d,%d,%s,%s' % (n, h, w, device, channel_first)
    if str_id not in mesh_grid_cache:
        x_base = torch.arange(0, w, dtype=torch.float32, device=device)[None, None, :].expand(n, h, w)
        y_base = torch.arange(0, h, dtype=torch.float32, device=device)[None, None, :].expand(n, w, h)  # NWH
        grid = torch.stack([x_base, y_base.transpose(1, 2)], 1)  # B2HW
        if not channel_first:
            grid = grid.permute(0, 2, 3, 1)  # BHW2
        mesh_grid_cache[str_id] = grid
    return mesh_grid_cache[str_id]


def squared_distance(xyz1: torch.Tensor, xyz2: torch.Tensor):
    """
    Calculate the Euclidean squared distance between every two points.
    :param xyz1: the 1st set of points, [batch_size, n_points_1, 3]
    :param xyz2: the 2nd set of points, [batch_size, n_points_2, 3]
    :return: squared distance between every two points, [batch_size, n_points_1, n_points_2]
    """
    assert xyz1.shape[-1] == xyz2.shape[-1] and xyz1.shape[-1] <= 3  # assert channel_last
    batch_size, n_points1, n_points2 = xyz1.shape[0], xyz1.shape[1], xyz2.shape[1]
    dist = -2 * torch.matmul(xyz1, xyz2.permute(0, 2, 1))
    dist += torch.sum(xyz1 ** 2, -1).view(batch_size, n_points1, 1)
    dist += torch.sum(xyz2 ** 2, -1).view(batch_size, 1, n_points2)
    return dist


def k_nearest_neighbor(input_xyz: torch.Tensor, query_xyz: torch.Tensor, k: int):
    """
    Calculate k-nearest neighbor for each query.
    :param input_xyz: a set of points, [batch_size, n_points, 3] or [batch_size, 3, n_points]
    :param query_xyz: a set of centroids, [batch_size, n_queries, 3] or [batch_size, 3, n_queries]
    :param k: int
    :param cpp_impl: whether to use the CUDA C++ implementation of k-nearest-neighbor
    :return: indices of k-nearest neighbors, [batch_size, n_queries, k]
    """
    if input_xyz.shape[1] <= 3:  # channel_first to channel_last
        assert query_xyz.shape[1] == input_xyz.shape[1]
        input_xyz = input_xyz.transpose(1, 2).contiguous()
        query_xyz = query_xyz.transpose(1, 2).contiguous()
    
    dists = squared_distance(query_xyz, input_xyz)
    return dists.topk(k, dim=2, largest=False).indices.to(torch.long)


def grid_sample_wrapper(feat_2d, xy):
    image_h, image_w = feat_2d.shape[2:]
    new_x = 2.0 * xy[:, 0] / (image_w - 1) - 1.0  # [bs, n_points] [-1,1]
    new_y = 2.0 * xy[:, 1] / (image_h - 1) - 1.0  # [bs, n_points]
    new_xy = torch.cat([new_x[:, :, None, None], new_y[:, :, None, None]], dim=-1)  # [bs, n_points, 1, 2]
    result = grid_sample(feat_2d, new_xy, 'bilinear', align_corners=True)  # [bs, n_channels, n_points, 1]
    return result[..., 0]


