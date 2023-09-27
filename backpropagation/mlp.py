import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # First linear transformation
        z1 = torch.matmul(x, self.parameters['W1'].T) + self.parameters['b1']

        # First activation function
        if self.f_function == 'relu':
            a1 = torch.relu(z1)
        elif self.f_function == 'sigmoid':
            a1 = torch.sigmoid(z1)
        else:
            a1 = z1

        # Second linear transformation
        z2 = torch.matmul(a1, self.parameters['W2'].T) + self.parameters['b2']

        # Second activation function
        if self.g_function == 'relu':
            a2 = torch.relu(z2)
        elif self.g_function == 'sigmoid':
            a2 = torch.sigmoid(z2)
        else:
            a2 = z2

        # Cache intermediate values for backward pass
        self.cache['x'] = x
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        return a2

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        x = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        z2 = self.cache['z2']
        a2 = self.cache['a2']

        # Calculate gradients for second linear layer and activation function
        if self.g_function == 'relu':
            dz2 = dJdy_hat * (a2 > 0).float()
        elif self.g_function == 'sigmoid':
            dz2 = dJdy_hat * (a2 * (1 - a2))
        else:
            dz2 = dJdy_hat

        self.grads['dJdW2'] = torch.matmul(dz2.T, a1)
        self.grads['dJdb2'] = torch.sum(dz2, dim=0)

        # Calculate gradients for first linear layer and activation function
        dz1 = torch.matmul(dz2, self.parameters['W2']) 
        if self.f_function == 'relu':
            dz1 = dz1 * (z1 > 0).float()  
        elif self.f_function == 'sigmoid':
            dz1 = dz1 * (a1 * (1 - a1)) 

        self.grads['dJdW1'] = torch.matmul(dz1.T, x)
        self.grads['dJdb1'] = torch.sum(dz1, dim=0)

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    J = torch.mean((y - y_hat) ** 2) 
    dJdy_hat = -2 * (y - y_hat) / torch.numel(y)
    return J, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    epsilon = 1e-12
    J = -torch.mean(y * torch.log(y_hat + epsilon) + (1 - y) * torch.log(1 - y_hat + epsilon)) 
    dJdy_hat = -(y / (y_hat + epsilon) - (1 - y) / (1 - y_hat + epsilon)) / torch.numel(y)
    return J, dJdy_hat
