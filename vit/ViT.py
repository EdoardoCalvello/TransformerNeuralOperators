import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, encoder_depth, num_heads, mlp_dim, dropout):
        super(ViT, self).__init__()

        # If you write a flexible method to chop into patches, or write the EC conv method and pass
        #  the correct optional arument to Conv2d / Conv1d, you don't need to check this:
        assert image_size % patch_size == 0, "image size must be divisible by patch size"
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # Patches
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2 # RGB has 3 channels
        
        # Layers
        self.patch_embedding = nn.Linear(self.patch_dim, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=mlp_dim,
                                       dropout=dropout, activation='gelu', batch_first=True),
            num_layers=encoder_depth
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.Tanh(),
            nn.Linear(mlp_dim, num_classes)
        )

    def chop_to_patches(self, x, b, c):
        ''' Chops each image in batch x into patches of dimensions patch_size**2.
            This method uses tensor reshaping.
            input:
              x: batch of images, with shape (batch_size, num_channels, height, width)
              b: batch_size
              c: num_channels
            output:
              chopped_x: chop-chop! shape (batch_size, num_patch_per_row, num_patch_per_col,
                                           num_channels, patch_size, patch_size)
        '''
        # - shape is now (batch_size, num_channels, num_patch_per_row, patch_size, num_patch_per_col, patch_size)
        chopped_x = x.reshape(b, c, self.image_size // self.patch_size, self.patch_size,
                      self.image_size // self.patch_size, self.patch_size)
        
        # - shape is now (batch_size, num_patch_per_row, num_patch_per_col, num_channels, patch_size, patch_size)
        # TODO: Fill-in the code to obtained the desired shape as mentioned above
        chopped_x = torch.permute(chopped_x, (0,2,4,1,3,5))

        return chopped_x

    def chop_to_patches_with_conv(self, x, b, c):
        ''' Chops each image in batch x into patches of dimensions patch_size**2.
            This method uses convolutional filters.
            input:
              x: batch of images, with shape (batch_size, num_channels, height, width)
              b: batch_size
              c: num_channels
            output:
              chopped_x: chop-chop! shape (batch_size, num_patch_per_row, num_patch_per_col,
                                           num_channels, patch_size, patch_size)
        '''
        # [extra-credit] TODO: Implement a method to chop images into patches using a conv kernel
        chopped_x = None
        return chopped_x


    def forward(self, x):
        # shape is (batch_size, num_channels, height, width)
        b, c, w, h = x.shape
        assert h == self.image_size and w == self.image_size, \
            f"Input image size ({h}*{w}) doesn't match model expected size ({self.image_size}*{self.image_size})"

        # Chop into patches
        x = self.chop_to_patches(x, b, c)

        # Flatten patches - recall patch_dim=(num_channels*patch_size**2) 
        x = x.reshape(b, self.num_patches, self.patch_dim)

        # Linear embedding
        # TODO: Call the appropriate method to perform linear embedding
        x = self.patch_embedding(x)

        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position tokens
        x += self.position_embedding[:, :(x.shape[1])]

        # Transformer encoding
        x = self.transformer_encoder(x)

        # MLP head
        cls_token = x[:, 0, :]
        x = self.mlp_head(cls_token)
        return x