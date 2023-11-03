import numpy as np
import torch
import matplotlib.pyplot as plt


def test1(model, device, test_loader):
    model.eval().to(device)
    masks3d = []
    for batch_idx, (data, _, originalSize) in enumerate(test_loader):
        data = data.to(device).type(torch.float32)
        y_p = model(data)

        masks2d = np.array(torch.Tensor.cpu(y_p))
        masks3d.append(masks2d)
    masks3d = np.array(masks3d, dtype=np.float32)

    return masks3d


def test(model, device, test_loader):
    model.eval().to(device)
    masks3d = []
    ideal_masks = []
    for batch_idx, (data, lab, originalSize) in enumerate(test_loader):
        data = data.to(device).type(torch.float32)
        y_p = model(data)
        masks2d = np.array(torch.Tensor.cpu(y_p))
        
        # Reconstruct initial length for each mask:
        masks2d = masks2d.reshape(masks2d.shape[0]*masks2d.shape[1], masks2d.shape[2])

        # print("Truncated shape", np.shape(masks2d[:originalSize]))
        masks2d = (np.array(masks2d[:originalSize])).T
        # ideal = np.squeeze((np.array(np.array(torch.Tensor.cpu(lab)))))
        # ideal = ideal[:originalSize].T
        # Append all masks to list
        masks3d.append(masks2d)
        # ideal_masks.append(ideal)
        # print("Estimated mask dims: ", np.shape(masks2d))
        # print("Ideal mask dims: ", np.shape(ideal))

    # plt.figure()
    # plt.imshow(ideal_masks[10], origin='lower')
    # plt.savefig('./ideal_mask.png', dpi=600)
    # Convert each individual array within the list to a numpy array,
    # instead of trying to convert the entire list to a single numpy array
    # Can't convert this to numpy because there are different lengths for each file.
    # masks3d = np.array(masks3d)
    return masks3d

