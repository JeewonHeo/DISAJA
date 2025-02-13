from typing import Tuple
import torch
from torch import Tensor
from torch import nn

from disaja.modules import SelfAttentionBlock
from disaja.modules import DecoderBlock
from disaja.modules import ObjWise
from disaja.modules import ParticleFlowMerger
from disaja.modules import ConParticleFlowMerger
from disaja.modules import ScatterMean
from disaja.data.dataset import TTbarDileptonBatch
from disaja.data.dataset import ConBatch

from torch.nn.utils.rnn import pad_sequence


class TTbarDileptonSAJA(nn.Module):
    def __init__(self,
                 dim_jet: int,
                 dim_lepton: int,
                 dim_met: int,
                 dim_output: int,
                 dim_ffnn: int=256,
                 num_blocks: int=1,
                 num_heads: int=2,
                 depth: int=32,
                 dropout_rate: float=0.1,
    ) -> None:
        """
        Transformer-based model for TTbar Dilepton analysis.

        Args:
            dim_jet (int): Input feature dimension for jets.
            dim_lepton (int): Input feature dimension for leptons.
            dim_met (int): Input feature dimension for MET.
            dim_output (int): Output feature dimension.
            dim_ffnn (int): Feedforward neural network hidden dimension.
            num_blocks (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            depth (int): Dimension of each attention head.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.dim_jet = dim_jet
        self.dim_lepton = dim_lepton
        self.dim_met = dim_met
        self.dim_ffnn = dim_ffnn
        self.num_blocks = num_blocks
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dim_output = dim_output

        self.dim_model = num_heads * depth

        # input projectuon

        ## jet projection
        self.jet_projection = ObjWise(
            nn.Linear(dim_jet, dim_ffnn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.GELU())

        ## lepton projection
        self.lepton_projection = nn.Sequential(
            nn.Linear(dim_lepton, dim_ffnn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.GELU())

        ## met projection
        self.met_projection = nn.Sequential(
            nn.Linear(dim_met, dim_ffnn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.GELU())

        # Particle flow merger to combine jets, leptons, and MET into a single sequence
        self.merger = ParticleFlowMerger()

        # Encoder attention blocks
        encoder_attention_blocks = []
        for _ in range(num_blocks):
            block = SelfAttentionBlock(self.dim_model, num_heads, dim_ffnn,
                                       dropout_rate)
            encoder_attention_blocks.append(block)
        self.encoder_attention_blocks = nn.ModuleList(encoder_attention_blocks)

        # Decoder attention blocks
        decoder_attention_blocks = []
        for _ in range(num_blocks):
            block = DecoderBlock(self.dim_model, num_heads, dim_ffnn,
                                 dropout_rate)
            decoder_attention_blocks.append(block)
        self.decoder_attention_blocks = nn.ModuleList(decoder_attention_blocks)

        # Final output projection
        self.output_projection = ObjWise(
            nn.Linear(self.dim_model, dim_ffnn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, dim_output, bias=True))

    def forward(self, batch: TTbarDileptonBatch) -> Tensor:
        """
        Forward pass of TTbarDileptonSAJA model.

        Args:
            batch (TTbarDileptonBatch): Input batch of TTbar dilepton event.

        Returns:
            Tensor: The model output tensor.
        """
        # input projection
        jet = self.jet_projection(input=batch.jet,
                                  data_mask=batch.jet_data_mask)

        lepton = self.lepton_projection(batch.lepton)

        met = self.met_projection(batch.met)

        # Merge inputs into a unified particle flow representation
        x, lengths, data_mask = self.merger(
            jet=jet,
            jet_lengths=batch.jet_lengths,
            jet_data_mask=batch.jet_data_mask,
            lepton=lepton,
            met=met)

        data_mask = data_mask.to(x.device)

        # Pass through encoder attention blocks
        for block in self.encoder_attention_blocks:
            x, attention = block(x, data_mask)

        # Decoder stage with attention over jets
        source = x
        x = jet
        for block in self.decoder_attention_blocks:
            x, attention1, attention2 = block(source, x, data_mask, batch.jet_data_mask)

        # Apply final output projection
        x = self.output_projection(x, batch.jet_data_mask)

        return x

class ConSAJA(nn.Module):
    def __init__(self,
                 dim_track: int,
                 dim_tower: int,
                 dim_output: int,
                 dim_ffnn: int=128,
                 num_blocks: int=6,
                 num_heads: int=10,
                 depth: int=32,
                 dropout_rate: float=0.1,
    ) -> None:
        """
        Transformer-based model for jet constituent analysis.

        Args:
            dim_track (int): Input feature dimension for tracks.
            dim_tower (int): Input feature dimension for calorimeter towers.
            dim_output (int): Output feature dimension.
            dim_ffnn (int): Feedforward neural network hidden dimension.
            num_blocks (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            depth (int): Dimension of each attention head.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.dim_track = dim_track
        self.dim_tower = dim_tower
        self.dim_output = dim_output
        self.dim_ffnn = dim_ffnn
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.depth = depth
        self.dropout_rate = dropout_rate

        self.dim_model = num_heads * depth

        # input projectuon

        ## track projection
        self.track_projection = ObjWise(
            nn.Linear(dim_track, dim_ffnn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.GELU())

        ## tower projection
        self.tower_projection = ObjWise(
            nn.Linear(dim_tower, dim_ffnn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.GELU())

        ## merger
        self.merger = ConParticleFlowMerger()

        # aggregate
        self.aggregation = ScatterMean()

        attention_blocks = []
        for _ in range(num_blocks):
            block = SelfAttentionBlock(self.dim_model, num_heads, dim_ffnn, dropout_rate)
            attention_blocks.append(block)
        self.attention_blocks = nn.ModuleList(attention_blocks)

        self.output_projection = nn.Sequential(
                nn.Linear(self.dim_model, dim_ffnn, bias=True),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(dim_ffnn, dim_output, bias=True))

    def forward(self, batch: ConBatch) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:
            mask:
        Returns:
        """
        # input projection
        track = self.track_projection(input=batch.track,
                                      data_mask=batch.track_data_mask)

        tower = self.tower_projection(input=batch.tower,
                                      data_mask=batch.tower_data_mask)

        x, lengths, data_mask = self.merger(
                track=track,
                track_lengths=batch.track_lengths,
                track_data_mask=batch.track_data_mask,
                tower=tower,
                tower_lengths=batch.tower_lengths,
                tower_data_mask=batch.tower_data_mask,
                )

        data_mask = data_mask.to(x.device)

        for block in self.attention_blocks:
            x, attention = block(x, data_mask)

        x = self.aggregation(
                x,
                data_mask,
                lengths,
                )

        return x


class TTDileptonWithConSAJA(nn.Module):
    def __init__(self,
                 dim_track: int,
                 dim_tower: int,
                 dim_lepton: int,
                 dim_met: int,
                 dim_output: int,
                 dim_ffnn: int=128,
                 num_blocks: int=6,
                 num_heads: int=10,
                 depth: int=32,
                 dropout_rate: float=0.1,
                 pretrained_model: str=None,
                 pretrained_model_freeze: bool=True
) -> None:
        """
        Transformer-based model for analyzing TTbar Dilepton events, integrating jet constituent
        information with event-level features.

        Args:
            dim_track (int): Feature dimension of track inputs.
            dim_tower (int): Feature dimension of tower inputs.
            dim_lepton (int): Feature dimension of lepton inputs.
            dim_met (int): Feature dimension of MET inputs.
            dim_output (int): Output feature dimension.
            dim_ffnn (int): Hidden dimension for feedforward layers.
            num_blocks (int): Number of transformer attention blocks.
            num_heads (int): Number of attention heads.
            depth (int): Dimension of each attention head.
            dropout_rate (float): Dropout probability.
            pretrained_model (str, optional): Path to the pretrained jet constituents model.
            pretrained_model_freeze (bool, optional): Whether to freeze the pretrained model.
        """
        super().__init__()

        self.dim_track = dim_track
        self.dim_tower = dim_tower
        self.dim_lepton = dim_lepton
        self.dim_met = dim_met
        self.dim_output = dim_output
        self.dim_ffnn = dim_ffnn
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.depth = depth
        self.dropout_rate = dropout_rate

        self.dim_model = num_heads * depth

        # Jet constituents encoder (processes track & tower features into jet representations)
        self.jet_constituents_encoder = ConSAJA(
                dim_track=dim_track,
                dim_tower=dim_tower,
                dim_output=3,
                dim_ffnn=256,
                num_blocks=2,
                num_heads=2,
                depth=32
        )

        if pretrained_model is not None:
            try:
                self.jet_constituents_encoder.load_state_dict(
                    torch.load(pretrained_model,
                               map_location=torch.device('cpu'))['model']
                )
            except Exception as err:
                print(err)

        if pretrained_model_freeze:
            for param in self.jet_constituents_encoder.parameters():
                param.requires_grad = False

        ## jet projection
        self.jet_projection = ObjWise(
            nn.Linear(64, dim_ffnn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.GELU())

        ## lepton projection
        self.lepton_projection = nn.Sequential(
            nn.Linear(dim_lepton, dim_ffnn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.GELU())

        ## met projection
        self.met_projection = nn.Sequential(
            nn.Linear(dim_met, dim_ffnn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.GELU())

        ## merger
        self.merger = ParticleFlowMerger()

        encoder_attention_blocks = []
        for _ in range(num_blocks):
            block = SelfAttentionBlock(self.dim_model, num_heads, dim_ffnn,
                                       dropout_rate)
            encoder_attention_blocks.append(block)
        self.encoder_attention_blocks = nn.ModuleList(encoder_attention_blocks)

        decoder_attention_blocks = []
        for _ in range(num_blocks):
            block = DecoderBlock(self.dim_model, num_heads, dim_ffnn,
                                 dropout_rate)
            decoder_attention_blocks.append(block)
        self.decoder_attention_blocks = nn.ModuleList(decoder_attention_blocks)

        self.output_projection = ObjWise(
            nn.Linear(self.dim_model, dim_ffnn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, dim_output, bias=True))

    def forward(self, batch: (TTDileptonWithConBatch, ConBatch)) -> Tensor:
        event_batch = batch[0]
        jet_batch = batch[1]

        jet = self.jet_constituents_encoder(jet_batch)

        # Merge jets per event using batch indices
        seq_idxs = event_batch.batch_idx
        batch_size = len(event_batch.jet_lengths)
        jet = pad_sequence(
            [jet[seq_idxs == idx] for idx in range(batch_size)],
            batch_first=True)

        # Project the merged jet representations into the model space
        jet = self.jet_projection(input=jet,
                                  data_mask=event_batch.jet_data_mask)

        # Project leptons and MET features
        lepton = self.lepton_projection(event_batch.lepton)
        met = self.met_projection(event_batch.met)

        # Merge all features into a single particle flow sequence
        x, lengths, data_mask = self.merger(
            jet=jet,
            jet_lengths=event_batch.jet_lengths,
            jet_data_mask=event_batch.jet_data_mask,
            lepton=lepton,
            met=met)

        data_mask = data_mask.to(x.device)

        # Pass through encoder attention blocks
        for block in self.encoder_attention_blocks:
            x, attention = block(x, data_mask)

        # Decoder stage (processes jets with learned features from the encoder)
        source = x
        x = jet
        for block in self.decoder_attention_blocks:
            x, attention1, attention2 = block(source,
                                              x,
                                              data_mask,
                                              event_batch.jet_data_mask
                                              )

        # Final projection to get the model output
        x = self.output_projection(x, event_batch.jet_data_mask)

        return x
