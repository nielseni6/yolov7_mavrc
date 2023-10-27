import torch


def generate_attribution_map_vanilla_grad(model, input_tensor, norm=False):
    # Set model to evaluation mode
    model.eval()

    # Set requires_grad attribute of tensor. Important for computing gradients
    input_tensor.requires_grad = True

    # Forward pass
    output = model(input_tensor)

    # Find index of class with highest score
    _, index = torch.max(output, 1)

    # Zero gradients
    model.zero_grad()

    # Calculate gradients of output with respect to input
    output[0, index].backward()

    # Get gradients
    gradients = input_tensor.grad.data

    # Convert gradients to numpy array
    gradients = gradients.detach().numpy()[0]

    if norm:
        # Take absolute values of gradients
        gradients = np.absolute(gradients)

        # Sum across color channels
        attribution_map = np.sum(gradients, axis=0)

        # Normalize attribution map
        attribution_map /= np.max(attribution_map)
    else:
        # Sum across color channels
        attribution_map = gradients

    return attribution_map
