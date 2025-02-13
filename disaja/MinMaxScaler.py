import torch


# MinMaxScaler for normalizing jet, lepton, and MET data
class MinMaxScaler:
    def __init__(self, jet_branches, lep_branches, met_branches):
        self.jet_dataset_min = torch.full((len(jet_branches), ), fill_value=1e6)
        self.jet_dataset_max = torch.full((len(jet_branches), ), fill_value=-1e6)
        self.lep_dataset_min = torch.full((len(lep_branches), ), fill_value=1e6)
        self.lep_dataset_max = torch.full((len(lep_branches), ), fill_value=-1e6)
        self.met_dataset_min = torch.full((len(met_branches), ), fill_value=1e6)
        self.met_dataset_max = torch.full((len(met_branches), ), fill_value=-1e6)

    def fit(self, dataset):
        """
        Compute the minimum and maximum values for each feature in the dataset.
        """
        for data in dataset:
            jet = data.jet
            lep = data.lepton
            met = data.met

            # Compute min and max for each feature
            jet_min = jet.min(axis=0)[0]
            jet_max = jet.max(axis=0)[0]
            lep_min = lep.min(axis=0)[0]
            lep_max = lep.max(axis=0)[0]

            # Update global min and max values using element-wise comparison
            self.jet_dataset_min = torch.where(self.jet_dataset_min > jet_min, jet_min, self.jet_dataset_min)
            self.jet_dataset_max = torch.where(self.jet_dataset_max < jet_max, jet_max, self.jet_dataset_max)
            self.lep_dataset_min = torch.where(self.lep_dataset_min > lep_min, lep_min, self.lep_dataset_min)
            self.lep_dataset_max = torch.where(self.lep_dataset_max < lep_max, lep_max, self.lep_dataset_max)
            self.met_dataset_min = torch.where(self.met_dataset_min > met, met, self.met_dataset_min)
            self.met_dataset_max = torch.where(self.met_dataset_max < met, met, self.met_dataset_max)

    def transform(self, dataset):
        """
        Normalize dataset using Min-Max scaling.
        """
        # Compute scaling factors
        jet_shift = self.jet_dataset_min
        jet_scale = (self.jet_dataset_max - self.jet_dataset_min)
        lep_shift = self.lep_dataset_min
        lep_scale = self.lep_dataset_max - self.lep_dataset_min
        met_shift = self.met_dataset_min
        met_scale = self.met_dataset_max - self.met_dataset_min

        # Apply normalization
        for data in dataset:
            jet = data.jet
            lep = data.lepton
            met = data.met
            jet.sub_(jet_shift).div_(jet_scale)
            lep.sub_(lep_shift).div_(lep_scale)
            met.sub_(met_shift).div_(met_scale)


# MinMaxScaler for track and tower data (continuous input)
class ConMinMaxScaler:
    def __init__(self, track_branches, tower_branches):
        # Initialize min and max tensors with extreme values
        self.track_dataset_min = torch.full((len(track_branches), ), fill_value=1e6)
        self.track_dataset_max = torch.full((len(track_branches), ), fill_value=-1e6)
        self.tower_dataset_min = torch.full((len(tower_branches), ), fill_value=1e6)
        self.tower_dataset_max = torch.full((len(tower_branches), ), fill_value=-1e6)

    def fit(self, dataset):
        """
        Compute the minimum and maximum values for track and tower features.
        """
        for data in dataset:
            track = data.track
            tower = data.tower

            # Compute min and max values
            track_min = track.min(axis=0)[0]
            track_max = track.max(axis=0)[0]
            tower_min = tower.min(axis=0)[0]
            tower_max = tower.max(axis=0)[0]

            # Update global min and max values
            self.track_dataset_min = torch.where(self.track_dataset_min > track_min, track_min, self.track_dataset_min)
            self.track_dataset_max = torch.where(self.track_dataset_max < track_max, track_max, self.track_dataset_max)
            self.tower_dataset_min = torch.where(self.tower_dataset_min > tower_min, tower_min, self.tower_dataset_min)
            self.tower_dataset_max = torch.where(self.tower_dataset_max < tower_max, tower_max, self.tower_dataset_max)

    def transform(self, dataset):
        """
        Normalize dataset using Min-Max scaling.
        """
        track_shift = self.track_dataset_min
        track_scale = self.track_dataset_max - self.track_dataset_min
        tower_shift = self.tower_dataset_min
        tower_scale = self.tower_dataset_max - self.tower_dataset_min

        # Apply normalization
        for data in dataset:
            track = data.track
            tower = data.tower
            track.sub_(track_shift).div_(track_scale)
            tower.sub_(tower_shift).div_(tower_scale)


# Combined MinMaxScaler for track, tower, lepton, and MET data
class TTWithConMinMaxScaler:
    def __init__(self, track_branches, tower_branches, lep_branches, met_branches):
        self.track_dataset_min = torch.full((len(track_branches), ), fill_value=1e6)
        self.track_dataset_max = torch.full((len(track_branches), ), fill_value=-1e6)
        self.tower_dataset_min = torch.full((len(tower_branches), ), fill_value=1e6)
        self.tower_dataset_max = torch.full((len(tower_branches), ), fill_value=-1e6)
        self.lep_dataset_min = torch.full((len(lep_branches), ), fill_value=1e6)
        self.lep_dataset_max = torch.full((len(lep_branches), ), fill_value=-1e6)
        self.met_dataset_min = torch.full((len(met_branches), ), fill_value=1e6)
        self.met_dataset_max = torch.full((len(met_branches), ), fill_value=-1e6)

    def fit(self, dataset):
        """
        Compute the minimum and maximum values for each feature across the dataset.
        """
        for data in dataset:
            # Concatenate track and tower data along the batch dimension
            track = torch.cat(data.track)
            tower = torch.cat(data.tower)
            lep = data.lepton
            met = data.met

            # Compute min and max values
            track_min = track.min(axis=0)[0]
            track_max = track.max(axis=0)[0]
            tower_min = tower.min(axis=0)[0]
            tower_max = tower.max(axis=0)[0]
            lep_min = lep.min(axis=0)[0]
            lep_max = lep.max(axis=0)[0]

            # Update global min and max values
            self.track_dataset_min = torch.where(self.track_dataset_min > track_min, track_min, self.track_dataset_min)
            self.track_dataset_max = torch.where(self.track_dataset_max < track_max, track_max, self.track_dataset_max)
            self.tower_dataset_min = torch.where(self.tower_dataset_min > tower_min, tower_min, self.tower_dataset_min)
            self.tower_dataset_max = torch.where(self.tower_dataset_max < tower_max, tower_max, self.tower_dataset_max)
            self.lep_dataset_min = torch.where(self.lep_dataset_min > lep_min, lep_min, self.lep_dataset_min)
            self.lep_dataset_max = torch.where(self.lep_dataset_max < lep_max, lep_max, self.lep_dataset_max)
            self.met_dataset_min = torch.where(self.met_dataset_min > met, met, self.met_dataset_min)
            self.met_dataset_max = torch.where(self.met_dataset_max < met, met, self.met_dataset_max)

    def transform(self, dataset):
        """
        Normalize dataset using Min-Max scaling.
        """
        track_shift = self.track_dataset_min
        track_scale = self.track_dataset_max - self.track_dataset_min
        tower_shift = self.tower_dataset_min
        tower_scale = self.tower_dataset_max - self.tower_dataset_min
        lep_shift = self.lep_dataset_min
        lep_scale = self.lep_dataset_max - self.lep_dataset_min
        met_shift = self.met_dataset_min
        met_scale = self.met_dataset_max - self.met_dataset_min

        for data in dataset:
            # Normalize each feature
            track = torch.cat(data.track)
            tower = torch.cat(data.tower)
            for track in data.track:
                track.sub_(track_shift).div_(track_scale)
            for tower in data.tower:
                tower.sub_(tower_shift).div_(tower_scale)
            lep = data.lepton
            met = data.met
            lep.sub_(lep_shift).div_(lep_scale)
            met.sub_(met_shift).div_(met_scale)

