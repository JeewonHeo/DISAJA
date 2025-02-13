import dataclasses
from typing import Union, List, Dict

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import tqdm
import uproot

from disaja.data.utils import TensorCollection, make_data_mask


@dataclasses.dataclass
class Batch:
    data: Tensor
    target: Tensor
    length: Tensor
    mask: Tensor

    def to(self, device):
        return Batch(*[each.to(device) for each in dataclasses.astuple(self)])


# TODO typing
class JetPartonAssignmentDataset(Dataset):

    def __init__(self,
                 path: Union[str, List[str]],
                 treepath: str,
                 data_branches: List[str],
                 target_branch: str,
                 num_workers: int = 1,
                 step_size: str = '64 MB',
    ) -> None:
        """
        Args:
        """
        self.data_branches = data_branches
        self.target_branch = target_branch

        if num_workers == 1:
            file_handler = uproot.MemmapSource
        elif num_workers > 1:
            file_handler = uproot.MultithreadedFileSource
        else:
            raise ValueError(f'num_workers={num_workers}')


        files = self._to_files(path, treepath)
        branches = data_branches + [target_branch]

        tree_iter = uproot.iterate(files,
                                   expressions=branches,
                                   library='np',
                                   report=True,
                                   step_size=step_size,
                                   num_workers=num_workers,
                                   file_handler=file_handler)

        # TODO verbose
        cls_name = self.__class__.__name__
        total = sum(each for _, _, each in uproot.num_entries(files))

        def set_description(pbar, done):
            progress = 100 * done / total
            description = f'[{cls_name}] {done} / {total} ({progress:.2f} %)'
            pbar.set_description(description)

        self._examples = []

        pbar = tqdm.tqdm(tree_iter)
        set_description(pbar, 0)
        for chunk, report in pbar:
            self._examples += self._process(chunk)
            set_description(pbar, report.stop)

    def _to_files(self, files, treepath):
        if isinstance(files, str):
            files = {files: treepath}
        elif isinstance(files, list):
            files = {each: treepath for each in files}
        else:
            raise TypeError
        return files

    def _process(self, chunk):
        data_chunk = [chunk[branch] for branch in self.data_branches]
        data_chunk = zip(*data_chunk)
        data_chunk = [np.stack(each, axis=1) for each in data_chunk]
        data_chunk = [each.astype(np.float32) for each in data_chunk]
        data_chunk = [torch.from_numpy(each) for each in data_chunk]

        target_chunk = chunk[self.target_branch]
        target_chunk = [each.astype(np.int64) for each in target_chunk]
        target_chunk = [torch.from_numpy(each) for each in target_chunk]

        example_chunk = list(zip(data_chunk, target_chunk))
        return example_chunk

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]

    @classmethod
    def collate(cls, batch):
        data, target = list(zip(*batch))
        length = torch.LongTensor([each.size(0) for each in data])
        data = pad_sequence(data, batch_first=True)
        mask = make_data_mask(data, length)
        target = pad_sequence(target, batch_first=True)
        return Batch(data, target, length, mask)



@dataclasses.dataclass(repr=False)
class TTbarDileptonExample(TensorCollection):
    r"""
    Args:
        jet: sequential, varialbe length, numerical
        lepton: sequential, fixed length (=2), numerical
        met: numerical
        target: sequential, variable length, categorical
    Shape:
        jet: (L_jet, H_jet)
        lepton: (2, H_lep)
        met: (H_met, )
        target: (L_jet, )
    """
    jet: Tensor
    lepton: Tensor
    met: Tensor
    target: Tensor
    reco: Tensor


@dataclasses.dataclass(repr=False)
class TTbarDileptonBatch(TensorCollection):
    jet: Tensor
    jet_lengths: Tensor
    jet_data_mask: Tensor
    lepton: Tensor
    met: Tensor
    target: Tensor
    reco: Tensor


class TTbarDileptonDataset(JetPartonAssignmentDataset):

    def __init__(self,
                 path: Union[str, List[str]],
                 treepath: str,
                 jet_branches: List[str],
                 lep_branches: List[str],
                 met_branches: List[str],
                 target_branch: str,
                 reco_branches: List[str],
                 num_workers: int = 1,
                 ) -> None:

        self.jet_branches = jet_branches
        self.lepton_branches = lep_branches
        self.met_branches = met_branches
        self.reco_branches = reco_branches

        self.target_branch = target_branch

        self.data_branches = self.jet_branches + self.lepton_branches + self.met_branches + self.reco_branches

        super().__init__(path=path,
                         treepath=treepath,
                         data_branches=self.data_branches,
                         target_branch=self.target_branch,
                         num_workers=num_workers)

    def process_numerical_sequence(self,
                                   chunk: Dict[str, np.ndarray],
                                   branches: List[str]
                                   ) -> List[Tensor]:
        obj_chunk = [chunk[branch] for branch in branches]
        obj_chunk = zip(*obj_chunk)
        obj_chunk = [np.stack(each, axis=1) for each in obj_chunk]
        obj_chunk = [torch.from_numpy(each).float() for each in obj_chunk]
        return obj_chunk

    def process_numerical(self,
                          chunk: Dict[str, np.ndarray],
                          branches: List[str]
                          ) -> List[Tensor]:
        obj_chunk = [chunk[branch] for branch in branches]
        obj_chunk = zip(*obj_chunk)
        obj_chunk = [torch.tensor(each).float() for each in obj_chunk]
        return obj_chunk

    def process_categorical_sequence(self,
                                     chunk: Dict[str, np.ndarray],
                                     branch: str
                                     ) -> List[Tensor]:
        obj_chunk = [torch.from_numpy(each).long() for each in chunk[branch]]
        return obj_chunk

    def _process(self, chunk):
        # jets
        jet_chunk = self.process_numerical_sequence(chunk, self.jet_branches)

        # lepton
        lepton_chunk = self.process_numerical_sequence(chunk, self.lepton_branches)

        # met
        met_chunk = self.process_numerical(chunk, self.met_branches)

        # target
        target_chunk = self.process_categorical_sequence(chunk, self.target_branch)

        # reco

        reco_chunk = self.process_numerical(chunk, self.reco_branches)

        # example
        example_chunk = zip(jet_chunk,
                            lepton_chunk,
                            met_chunk,
                            target_chunk,
                            reco_chunk)

        example_chunk = [TTbarDileptonExample(*each) for each in example_chunk]
        return example_chunk

    @classmethod
    def collate(cls, batch: List[TTbarDileptonExample]) -> TTbarDileptonBatch:
        batch = [dataclasses.astuple(example) for example in batch]
        jet, lepton, met, target, reco = zip(*batch)

        jet_lengths = torch.LongTensor([len(each) for each in jet])
        jet = pad_sequence(jet, batch_first=True)
        jet_data_mask = make_data_mask(jet, jet_lengths)

        lepton = torch.stack(lepton, dim=0)

        met = torch.stack(met, dim=0)

        target = pad_sequence(target, batch_first=True)

        reco = pad_sequence(reco, batch_first=True)

        batch = TTbarDileptonBatch(jet=jet,
                                   jet_lengths=jet_lengths,
                                   jet_data_mask=jet_data_mask,
                                   lepton=lepton,
                                   met=met,
                                   target=target,
                                   reco=reco)
        return batch


def make_data_mask_con(length):
    mask_shape = (len(length), max(length))
    data_mask = torch.full(size=mask_shape, fill_value=0, dtype=torch.bool)
    for m, l in zip(data_mask, length):
        m[: l].fill_(1)
    return data_mask

@dataclasses.dataclass(repr=False)
class ConExample(TensorCollection):
    track: Tensor
    tower: Tensor
    target: Tensor


@dataclasses.dataclass(repr=False)
class ConBatch(TensorCollection):
    track: Tensor
    track_lengths: Tensor
    track_data_mask: Tensor
    tower: Tensor
    tower_lengths: Tensor
    tower_data_mask: Tensor
    target: Tensor

class ConDataset(JetPartonAssignmentDataset):

    def __init__(self,
                 path: Union[str, List[str]],
                 treepath: str,
                 track_branches: List[str],
                 tower_branches: List[str],
                 target_branch: str,
                 num_workers: int = 1,
    ) -> None:

        self.track_branches = track_branches
        self.tower_branches = tower_branches

        self.target_branch = target_branch

        self.data_branches = self.track_branches + self.tower_branches

        super().__init__(path=path,
                         treepath=treepath,
                         data_branches=self.data_branches,
                         target_branch=self.target_branch,
                         num_workers=num_workers)

    def process_numerical_sequence(self,
                                   chunk: Dict[str, np.ndarray],
                                   branches: List[str]
    ) -> List[Tensor]:
        obj_chunk = [chunk[branch] for branch in branches]
        obj_chunk = zip(*obj_chunk)
        obj_chunk = [np.stack(each, axis=1) for each in obj_chunk]
        obj_chunk = [[np.array([f for f in zip(*j)]) for j in each]
                     for each in obj_chunk]
        obj_chunk = [[torch.from_numpy(jet) for jet in each]
                     for each in obj_chunk]
        obj_chunk = [jet.float() for each in obj_chunk for jet in each]
        obj_chunk = [torch.zeros(1, len(branches)) if len(each) == 0 else each
                     for each in obj_chunk]
        return obj_chunk

    def process_numerical(self,
                          chunk: Dict[str, np.ndarray],
                          branches: List[str]
    ) -> List[Tensor]:
        obj_chunk = [chunk[branch] for branch in branches]
        obj_chunk = zip(*obj_chunk)
        obj_chunk = [torch.tensor(each).float() for each in obj_chunk]
        return obj_chunk

    def process_categorical_sequence(self,
                                     chunk: Dict[str, np.ndarray],
                                     branch: str
    ) -> List[Tensor]:
        obj_chunk = [torch.tensor([1]).long()
                     if jet == 2 else torch.tensor([0]).long()
                     for each in chunk[branch]
                     for jet in each]
        obj_chunk = [torch.tensor([jet]).long()
                     for each in chunk[branch]
                     for jet in each]
        return obj_chunk

    def _process(self, chunk):
        # track
        track_chunk = self.process_numerical_sequence(chunk,
                                                      self.track_branches)

        # tower
        tower_chunk = self.process_numerical_sequence(chunk,
                                                      self.tower_branches)

        # target
        target_chunk = self.process_categorical_sequence(chunk,
                                                         self.target_branch)

        # example
        example_chunk = zip(track_chunk,
                            tower_chunk,
                            target_chunk)

        example_chunk = [ConExample(*each) for each in example_chunk]
        return example_chunk

    @classmethod
    def collate(cls, batch: List[ConExample]) -> ConBatch:
        batch = [dataclasses.astuple(example) for example in batch]
        track, tower, target = zip(*batch)

        track_lengths = torch.LongTensor([len(each) for each in track])
        track = pad_sequence(track, batch_first=True)
        track_data_mask = make_data_mask(track, track_lengths)

        tower_lengths = torch.LongTensor([len(each) for each in tower])
        tower = pad_sequence(tower, batch_first=True)
        tower_data_mask = make_data_mask(tower, tower_lengths)

        target = pad_sequence(target, batch_first=True)

        batch = ConBatch(
                track=track,
                track_lengths=track_lengths,
                track_data_mask=track_data_mask,
                tower=tower,
                tower_lengths=tower_lengths,
                tower_data_mask=tower_data_mask,
                target=target)
        return batch

@dataclasses.dataclass(repr=False)
class TTDileptonWithConExample(TensorCollection):
    track: Tensor
    tower: Tensor
    lepton: Tensor
    met: Tensor
    target: Tensor
    reco: Tensor


@dataclasses.dataclass(repr=False)
class TTDileptonWithConBatch(TensorCollection):
    track: Tensor
    track_lengths: Tensor
    track_data_mask: Tensor
    tower: Tensor
    tower_lengths: Tensor
    tower_data_mask: Tensor
    lepton: Tensor
    met: Tensor
    target: Tensor
    batch_idx: Tensor
    jet_lengths: Tensor
    jet_data_mask: Tensor
    reco: Tensor


class TTDileptonWithConDataset(JetPartonAssignmentDataset):

    def __init__(self,
                 path: Union[str, List[str]],
                 treepath: str,
                 track_branches: List[str],
                 tower_branches: List[str],
                 lepton_branches: List[str],
                 met_branches: List[str],
                 target_branch: str,
                 reco_branches: List[str],
                 num_workers: int = 1,
    ) -> None:
        self.track_branches = track_branches
        self.tower_branches = tower_branches
        self.lepton_branches = lepton_branches
        self.met_branches = met_branches

        self.target_branch = target_branch
        self.reco_branches = reco_branches

        self.data_branches = track_branches + tower_branches \
            + lepton_branches + met_branches + reco_branches

        super().__init__(path=path,
                         treepath=treepath,
                         data_branches=self.data_branches,
                         target_branch=self.target_branch,
                         num_workers=num_workers)

    def process_numerical_sequence(self,
                                   chunk: Dict[str, np.ndarray],
                                   branches: List[str]
    ) -> List[Tensor]:
        obj_chunk = [chunk[branch] for branch in branches]
        obj_chunk = zip(*obj_chunk)
        obj_chunk = [np.stack(np.array(each, dtype=object), axis=1) for each in obj_chunk]
        obj_chunk = [[torch.tensor([f for f in zip(*j)], dtype=torch.float) for j in each]
                     for each in obj_chunk]
        obj_chunk = [[torch.zeros(1, len(branches))
                      if len(jet) == 0 else jet for jet in each]
                     for each in obj_chunk]
        return obj_chunk

    def process_numerical_lepton(self,
                                 chunk: Dict[str, np.ndarray],
                                 branches: List[str]
    ) -> List[Tensor]:
        obj_chunk = [chunk[branch] for branch in branches]
        obj_chunk = zip(*obj_chunk)
        obj_chunk = [np.stack(each, axis=1) for each in obj_chunk]
        obj_chunk = [torch.from_numpy(each).float() for each in obj_chunk]
        return obj_chunk

    def process_numerical(self,
                          chunk: Dict[str, np.ndarray],
                          branches: List[str]
    ) -> List[Tensor]:
        obj_chunk = [chunk[branch] for branch in branches]
        obj_chunk = zip(*obj_chunk)
        obj_chunk = [torch.tensor(each).float() for each in obj_chunk]
        return obj_chunk

    def process_categorical_sequence(self,
                                     chunk: Dict[str, np.ndarray],
                                     branch: str
    ) -> List[Tensor]:
        obj_chunk = [torch.from_numpy(each).long() for each in chunk[branch]]
        return obj_chunk

    def _process(self, chunk):
        # track
        track_chunk = self.process_numerical_sequence(chunk,
                                                      self.track_branches)

        # tower
        tower_chunk = self.process_numerical_sequence(chunk,
                                                      self.tower_branches)

        # lepton
        lepton_chunk = self.process_numerical_lepton(chunk,
                                                     self.lepton_branches)

        # MET
        met_chunk = self.process_numerical(chunk, self.met_branches)

        # target
        target_chunk = self.process_categorical_sequence(chunk,
                                                         self.target_branch)

        reco_chunk = self.process_numerical(chunk, self.reco_branches)

        # example
        example_chunk = zip(track_chunk,
                            tower_chunk,
                            lepton_chunk,
                            met_chunk,
                            target_chunk,
                            reco_chunk,
                            )

        example_chunk = [TTDileptonWithConExample(*each)
                         for each in example_chunk]
        return example_chunk

    @classmethod
    def collate(cls,
                batch: List[TTDileptonWithConExample]
                ) -> (TTDileptonWithConBatch, ConBatch):
        batch = [dataclasses.astuple(example) for example in batch]
        track, tower, lepton, met, target, reco = zip(*batch)

        sub_track_lengths = torch.LongTensor(
            [len(jet) for event in track for jet in event]
        )
        sub_track = pad_sequence(
            [jet.clone().detach() for event in track for jet in event],
            batch_first=True
        )
        sub_track_data_mask = make_data_mask(sub_track, sub_track_lengths)

        sub_tower_lengths = torch.LongTensor(
            [len(jet) for event in tower for jet in event]
        )
        sub_tower = pad_sequence(
            [jet.clone().detach() for event in tower for jet in event],
            batch_first=True
        )
        sub_tower_data_mask = make_data_mask(sub_tower, sub_tower_lengths)

        sub_target = [torch.LongTensor([jet])
                      for event in target
                      for jet in event]

        track_lengths = [torch.LongTensor([len(jet) for jet in each])
                         for each in track]
        track = [pad_sequence(each, batch_first=True) for each in track]
        track_data_mask = [make_data_mask(trk, lengths)
                           for trk, lengths in zip(track, track_lengths)]

        tower_lengths = [torch.LongTensor([len(jet) for jet in each])
                         for each in tower]
        tower = [pad_sequence(each, batch_first=True) for each in tower]
        tower_data_mask = [make_data_mask(tow, lengths)
                           for tow, lengths in zip(tower, tower_lengths)]

        target = pad_sequence(target, batch_first=True)

        batch_idx = [torch.LongTensor([i for _ in range(len(each))])
                     for i, each in enumerate(track)]
        jet_lengths = torch.LongTensor([len(each) for each in batch_idx])
        jet_data_mask = make_data_mask_con(jet_lengths)
        batch_idx = torch.cat(batch_idx)

        lepton = torch.stack(lepton, dim=0)

        met = torch.stack(met, dim=0)

        reco = torch.stack(reco, dim=0)

        batch = TTDileptonWithConBatch(
            track=track,
            track_lengths=track_lengths,
            track_data_mask=track_data_mask,
            tower=tower,
            tower_lengths=tower_lengths,
            tower_data_mask=tower_data_mask,
            lepton=lepton,
            met=met,
            reco=reco,
            target=target,
            batch_idx=batch_idx,
            jet_lengths=jet_lengths,
            jet_data_mask=jet_data_mask
            )

        sub_batch = ConBatch(
            track=sub_track,
            track_lengths=sub_track_lengths,
            track_data_mask=sub_track_data_mask,
            tower=sub_tower,
            tower_lengths=sub_tower_lengths,
            tower_data_mask=sub_tower_data_mask,
            target=target
            )

        return batch, sub_batch
