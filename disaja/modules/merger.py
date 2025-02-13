import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple
from disaja.data.utils import make_data_mask


class ParticleFlowMerger(nn.Module):
    def forward(self,
                jet: Tensor,
                jet_lengths: Tensor,
                jet_data_mask: Tensor,
                lepton: Tensor,
                met: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Merges jet, lepton, and MET features into a single particle flow representation.
        
        Args:
            jet (Tensor): Tensor of jet features (batch_size, max_jet_count, jet_features).
            jet_lengths (Tensor): Number of valid jets per sample in the batch.
            jet_data_mask (Tensor): Boolean mask indicating valid jet positions.
            lepton (Tensor): Tensor of lepton features (batch_size, lepton_features).
            met (Tensor): Tensor of MET features (batch_size, met_features).
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - particle_flow (Tensor): Combined particle flow sequence (batch_size, max_seq_len, feature_dim).
                - lengths (Tensor): Tensor indicating the length of each sequence.
                - data_mask (Tensor): Boolean mask indicating valid positions in the padded sequence.
        """
        # Apply the mask to extract valid jet entries
        jet = [j[m] for j, m in zip(jet, jet_data_mask)]

        # add a fake sequence dimension
        # (batch, met_features) -> (batch, 1, met_features)
        met = met.unsqueeze(1)

        # Combine lepton, MET, and jet features into a single sequence per batch sample
        particle_flow = zip(lepton, met, jet)
        particle_flow = [torch.cat(each, dim=0) for each in particle_flow]

        # lengths = [len(each) for each in particle_flow]
        lengths = jet_lengths + 2 + 1 # jet_lengths + lepton_lengths + met_lengths

        # Pad sequences to match the longest sequence in the batch
        particle_flow = pad_sequence(particle_flow, batch_first=True)

        # Generate a mask indicating valid data positions
        data_mask = make_data_mask(particle_flow, lengths)

        return particle_flow, lengths, data_mask

class ConParticleFlowMerger(nn.Module):
    def forward(self,
                track: Tensor,
                track_lengths: Tensor,
                track_data_mask: Tensor,
                tower: Tensor,
                tower_lengths: Tensor,
                tower_data_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Merges track and tower features into a single particle flow representation.
        
        Args:
            track (Tensor): Tensor of track features (batch_size, max_track_count, track_features).
            track_lengths (Tensor): Number of valid tracks per sample in the batch.
            track_data_mask (Tensor): Boolean mask indicating valid track positions.
            tower (Tensor): Tensor of tower features (batch_size, max_tower_count, tower_features).
            tower_lengths (Tensor): Number of valid towers per sample in the batch.
            tower_data_mask (Tensor): Boolean mask indicating valid tower positions.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - particle_flow (Tensor): Combined particle flow sequence (batch_size, max_seq_len, feature_dim).
                - lengths (Tensor): Tensor indicating the length of each sequence.
                - data_mask (Tensor): Boolean mask indicating valid positions in the padded sequence.
        """

        # Apply masks to extract valid track and tower entries
        track = [t[m] for t, m in zip(track, track_data_mask)]
        tower = [t[m] for t, m in zip(tower, tower_data_mask)]

        # Combine track and tower features into a single sequence per batch sample
        particle_flow = zip(track, tower)
        particle_flow = [torch.cat(each, dim=0) for each in particle_flow]

        # Compute the sequence lengths: track_lengths + tower_lengths
        lengths = track_lengths + tower_lengths

        # Pad sequences to match the longest sequence in the batch
        particle_flow = pad_sequence(particle_flow, batch_first=True)

        # Generate a mask indicating valid data positions
        data_mask = make_data_mask(particle_flow, lengths)

        return particle_flow, lengths, data_mask
