import torch
import numpy as np



class Structure():
    def __init__(self, exist, observed, dataset):  #, shapes=None, example=None, names=None):
        """
        Stores metadata about tensor shapes and observedness. One of shapes or example (without batch dimension)
        must be provided to extract the shapes from.
        """
        self.exist = np.array(exist, dtype=np.uint8)
        self.observed = np.array([o for o, e in zip(observed, self.exist) if e], dtype=np.uint8)
        self.latent = 1 - self.observed
        example = dataset.__getitem__(0, will_augment=False)
        self.shapes = [t.shape for t, e in zip(example, self.exist) if e]
        self.is_onehot = [oh for oh, e in zip(dataset.is_onehot, self.exist) if e]
        names = dataset.names if hasattr(dataset, "names") else [f"tensor_{i}" for i in range(len(self.shapes))]
        self.names = [n for n, e in zip(names, self.exist) if e]
        print("Created structure with shapes", self.shapes, "and observedness", self.observed)
        # take graphical structure for GSDM
        if hasattr(dataset, "get_graphical_model_mask_and_indices"):
            self.graphical_model_mask_and_indices = dataset.get_graphical_model_mask_and_indices()
        if hasattr(dataset, "get_shareable_embedding_indices"):
            self.shareable_embedding_indices = dataset.get_shareable_embedding_indices()

    @property
    def latent_names(self):
        return [n for n, l in zip(self.names, self.latent) if l]

    @staticmethod
    def get_flattened(batch, select):
        return torch.cat([t.flatten(start_dim=1) for t, s in zip(batch, select) if s], dim=1)

    def flatten_batch(self, batch, contains_marg):
        if contains_marg:
            batch = [t for t, e in zip(batch, self.exist) if e]
        lats = self.get_flattened(batch, 1-self.observed)
        obs = tuple(t for t, o in zip(batch, self.observed) if o)
        return lats, obs

    def flatten_latents(self, batch, contains_marg):
        if contains_marg:
            batch = [t for t, e in zip(batch, self.exist) if e]
        return self.get_flattened(batch, 1-self.observed)

    def flatten_obs(self, batch, contains_marg):
        if contains_marg:
            batch = [t for t, e in zip(batch, self.exist) if e]
        return tuple(t for t, o in zip(batch, self.observed) if o)

    def unflatten_batch(self, lats, obs, pad_marg):
        data = []
        for shape, o in zip(self.shapes, self.observed):
            numel = np.prod(shape)
            if o:
                t, obs = obs[0], obs[1:]
            else:
                t, lats = lats[:, :numel], lats[:, numel:]
            data.append(t.reshape(-1, *shape))
        if pad_marg:
            data_iter = iter(data)
            data = [next(data_iter) if e else None for e in self.exist]
        return tuple(data)

    def unflatten_latents(self, lats):
        data = []
        for shape, l in zip(self.shapes, self.latent):
            if l:
                numel = np.prod(shape)
                t, lats = lats[:, :numel], lats[:, numel:]
                data.append(t.reshape(-1, *shape))
        return tuple(data)

    @property
    def latent_dim(self):
        return sum(np.prod(shape) for shape, o in zip(self.shapes, self.observed) if not o)


class StructuredArgument():
    def __init__(self, arg, structure, dtype=torch.float32): 
        # arg should be a list of scalars. If it is a single scalar, or list of length 1, it is broadcasted.
        if type(arg) in [int, float]:
            arg = (arg,)
        if len(arg) == 1:
            arg = arg * len(structure.exist)
        assert len(arg) == len(structure.exist)
        arg = [a for a, e in zip(arg, structure.exist) if e]  # filter to only existent tensors
        self.arg = tuple([a*torch.ones((1, *shape), dtype=dtype) for a, shape in zip(arg, structure.shapes)])
        self.structure = structure

    @property
    def lats(self):
        return self.structure.flatten_latents(self.arg, contains_marg=False)

    @property
    def obs(self):
        _, obs = self.structure.flatten_batch(self.arg, contains_marg=False)
        return obs
