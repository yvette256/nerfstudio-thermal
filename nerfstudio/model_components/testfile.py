import torch

"""
inputs: number of sample points, cube length (b-a) and spacing between sample and neighbor
output: sample points and their 6 neighbors, scaled based on cube length
"""

count = 0
all_densities = []


def sample_and_scale_points(num_points, a, b, spacing):
    """

    Args:
        num_points: number of samples
        a: aabb box[0]
        b: aabb box[1]
        spacing: distance between neighbor and sample point

    Returns:

    """
    sampled_points = torch.rand(num_points, 3)
    sampled_points = sampled_points.cuda()
    scaled_points = a[0][0] + (b[0][0] - a[0][0]) * sampled_points
    scaled_points = scaled_points.cuda()
    width = float((b[0][0] - a[0][0]) / spacing)
    nb1 = torch.Tensor((1, 0, 0))
    nb1 = nb1.cuda()
    nb2 = torch.Tensor((-1, 0, 0))
    nb2 = nb2.cuda()
    nb3 = torch.Tensor((0, 1, 0))
    nb3 = nb3.cuda()
    nb4 = torch.Tensor((0, -1, 0))
    nb4 = nb4.cuda()
    nb5 = torch.Tensor((0, 0, 1))
    nb5 = nb5.cuda()
    nb6 = torch.Tensor((0, 0, -1))
    nb6 = nb6.cuda()
    neighbor1 = torch.sub(scaled_points, nb1 * width, alpha=1)
    neighbor2 = torch.sub(scaled_points, nb2 * width, alpha=1)
    neighbor3 = torch.sub(scaled_points, nb3 * width, alpha=1)
    neighbor4 = torch.sub(scaled_points, nb4 * width, alpha=1)
    neighbor5 = torch.sub(scaled_points, nb5 * width, alpha=1)
    neighbor6 = torch.sub(scaled_points, nb6 * width, alpha=1)

    all_points = torch.vstack((scaled_points, neighbor1, neighbor2, neighbor3, neighbor4, neighbor5, neighbor6))
    all_points = all_points.cuda()
    return all_points
