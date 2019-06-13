"""Loss functions / Utilities"""

import torch.nn as nn


class Tacotron2Loss(nn.Module):
    """Tacotron2 Loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """Loss computation (Forward pass)
        """
        acoustics, gates = targets
        predicted_postnet_acoustics, predicted_acoustics, predicted_gates, _ = predictions

        acoustics.require_grad = False
        gates.require_grad = False

        gates = gates.view(-1, 1)
        predicted_gates = predicted_gates.view(-1, 1)

        # acoustic loss
        acoustic_loss = nn.MSELoss()(predicted_postnet_acoustics, acoustics) + \
            nn.MSELoss()(predicted_acoustics, acoustics)

        # gate loss
        gate_loss = nn.BCEWithLogitsLoss()(predicted_gates, gates)

        return acoustic_loss + gate_loss
