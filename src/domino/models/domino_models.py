#####################################################################
# domino_models.py
#####################################################################

from math import ceil
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from domino.models.layers import (
    ConvBlock,
    Residual,
)

class Base(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        
    def forward(self, x):
        return NotImplementedError("This method must be overridden by a subclass")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        
        # define schedulers
        if self.hparams.scheduler == "plateau":
            # LR plateaus when loss stops improving
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1),
                'monitor': 'val_loss'
            }
        elif self.hparams.scheduler == "cycle":
            # Cyclic scheduler cycles learning rate to encourage exploration and prevent stuck in local min
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                               base_lr=self.hparams.learning_rate / 2, 
                                                               max_lr=self.hparams.learning_rate * 2, 
                                                               cycle_momentum=False),
                'interval': 'step' # change lr after each batch                  
            }
        else:
            print('No scheduler used.')
            return [optimizer]
        
        return [optimizer], [lr_scheduler]
    

    def proc_batch(self, batch):
        # processes batch to return only tensor seq and target
         seq, y, boundary, chrom, start, end, direction = batch
         seq_input = seq.float()
         seq_input = torch.permute(seq_input, (0, 2, 1))
         y = y.float()
         boundary = boundary.float()
         return seq_input, y, boundary

    def training_step(self, batch, batch_idx):
        x, y, _ = self.proc_batch(batch)
        y_hat = self(x)

        print(y.shape, y_hat.shape) # make sure shape is correct

        # for scalar regression, need to be (bs, 1)
        y = torch.reshape(y, (len(y), 1))
        y_hat = torch.reshape(y_hat, (len(y_hat), 1))

        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx) -> None:
        x, y, boundary = self.proc_batch(batch)
        y_hat = self(x)

        y = torch.reshape(y, (len(y), 1))
        y_hat = torch.reshape(y_hat, (len(y_hat), 1))

        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

        boundary = torch.reshape(boundary, (len(boundary), 1))

        result = torch.cat([y_hat.cpu().detach(), y.cpu().detach(), boundary.cpu().detach()], dim=1)
        self.validation_step_outputs.append(result)

    def on_validation_epoch_end(self) -> None:
        all_val_result = torch.cat(self.validation_step_outputs, dim=0)

        print("all_val_result: ", all_val_result.shape)

        val_count_corr = stats.pearsonr(
            all_val_result[:, 0].flatten(), all_val_result[:, 1].flatten()
        )[0]
        self.log("val_corr", val_count_corr)

        # Positive sample MSE and correlation

        positive_val_result = all_val_result[all_val_result[:, 2] == 1]

        if positive_val_result.shape[0] == 0:
            positive_val_count_corr = 0
            positive_val_count_mse = 0
        else:
            positive_val_count_corr = stats.pearsonr(
                positive_val_result[:, 0].flatten(), positive_val_result[:, 1].flatten()
            )[0]
            positive_val_count_mse = F.mse_loss(positive_val_result[:, 0].flatten(), positive_val_result[:, 1].flatten())

        self.log("positive_val_corr", positive_val_count_corr)
        self.log("positive_val_mse", positive_val_count_mse)

        # Negative sample MSE and correlation

        negative_val_result = all_val_result[all_val_result[:, 2] == 0]

        if negative_val_result.shape[0] == 0:
            negative_val_count_corr = 0
            negative_val_count_mse = 0
        else:
            negative_val_count_corr = stats.pearsonr(
                negative_val_result[:, 0].flatten(), negative_val_result[:, 1].flatten()
            )[0]
            negative_val_count_mse = F.mse_loss(negative_val_result[:, 0].flatten(), negative_val_result[:, 1].flatten())

        self.log("negative_val_corr", negative_val_count_corr)
        self.log("negative_val_mse", negative_val_count_mse)

        self.validation_step_outputs.clear() # clear lists to prepare for next epoch

    def predict_step(self, batch, batch_idx):
        """
        Returns output of step.
        """
        x, y, boundary = self.proc_batch(batch)
        y_hat = self(x)
        y = torch.reshape(y, (len(y), 1))
        y_hat = torch.reshape(y_hat, (len(y_hat), 1))
        boundary = torch.reshape(boundary, (len(boundary), 1))
        return torch.cat([y_hat.cpu().detach(), y.cpu().detach(), boundary.cpu().detach()], dim=1)


class DominoSingleOutput(Base):
    def __init__(
        self, 
        context_length,
        boundary_length,
        ins_window_size, # in kb
        demean_context_bin_width,
        delta,
        model_random_seed=42,
        conv_layers=4,
        conv_repeat=1,
        kernel_number=512,
        kernel_length=19,
        filter_number=256,
        filter_size=3,
        pooling_size=2,
        h_layers=2,
        hidden_size=256,
        dropout=0.2,
        learning_rate=1e-4,
        weight_decay=0,
        scheduler='cycle',
        activation='gelu'
    ):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(model_random_seed)

        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activations.keys())}")
        self.activation = activations[activation]

        # has a bunch of 4 by kernel_number matrices that capture motifs (PWM)
        # ReLU activation decides you get motif hit and also the magnitude
        self.conv0 = ConvBlock(
            in_channels=4,
            out_channels=kernel_number,
            kernel_size=kernel_length,
            padding='same',
            activation=activation
        )

        # second conv block captures motif patterns
        # kernel size is 3 which means captures 
        # whether motifs are present
        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                in_channels=kernel_number,
                out_channels=filter_number,
                kernel_size=filter_size,
                padding='same',
                activation=activation
            )
        )

        fc_dim = 2 * context_length + boundary_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            in_channels=filter_number,
                            out_channels=filter_number,
                            kernel_size=filter_size,
                            padding='same',
                            activation=activation
                        )
                    )
                )
            # picks average of two adjacent cells for each channel
            # self.convlayers.append(nn.MaxPool1d(pooling_size))
            self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=False))

            # update fc_dim
            fc_dim = fc_dim // pooling_size

        extra_dim = fc_dim - boundary_length
        print('fc dim', fc_dim)
        print('extra dim', extra_dim)

        # at this point, DNA reduced to features of size fc_dim
        self.convlayers.append(
            Residual(
                ConvBlock(
                    in_channels=filter_number,
                    out_channels=filter_number,
                    kernel_size=filter_size,
                    padding='same',
                    activation=activation
                )
            )
        )

        # reduce fc_dim to size boundary_length (middle bin)
        # self.convlayers.append(nn.MaxPool1d(kernel_size=extra_dim + 1, stride=1)) # The output should be (B, c, boundary_length)
        if extra_dim > 0:
            self.convlayers.append(nn.AvgPool1d(kernel_size=extra_dim + 1, stride=1)) # The output should be (B, c, boundary_length)
        self.convlayers.append( # The output should be (B, 1, boundary_length), need to flatten to (B, boundary_length)
            ConvBlock(
                in_channels=filter_number,
                out_channels=1,
                kernel_size=1,
                padding='same',
                activation=activation
            )
        )

        # catch corner case
        if extra_dim > 0:
            self.fc0 = nn.Sequential(nn.Linear(boundary_length, hidden_size), self.activation)
        else:
            self.fc0 = nn.Sequential(nn.Linear(fc_dim, hidden_size), self.activation)
        
        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), self.activation, nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        self.out = nn.Linear(hidden_size, 1)

        # for capturing embeddings
        self.embeddings = None
    
    def forward(self, x):
        # x: (bs, 4, input_length)
        
        x = self.conv0(x)
        
        for layer in self.convlayers:
            x = layer(x)

        x = x.squeeze(1)
        x = self.fc0(x)
        for layer in self.fclayers:
            x = layer(x)
        x = self.out(x)
        
        return x
    
    def get_embeddings(self):
        return self.embeddings

    def _get_layer_hook(self):
        # Hook function that will be triggered during the forward pass
        def hook_fn(module, input, output):
            self.embeddings = output
        return hook_fn
    