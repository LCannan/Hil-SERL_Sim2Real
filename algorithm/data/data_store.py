from threading import Lock

import gymnasium as gym
from algorithm.data.replay_buffer import ReplayBuffer
from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        device: str = "cpu",
        include_grasp_penalty: bool = False,
    ):
        # `include_grasp_penalty` has to be forwarded, not just accepted: the
        # buffer allocates its columns from these flags, and `_insert_recursively`
        # iterates the *buffer's* keys -- so a transition carrying an unallocated
        # key is silently dropped rather than raising.
        ReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            include_grasp_penalty=include_grasp_penalty,
            device=device,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    def latest_data_id(self) -> int:
        return self._insert_index

    def get_latest_data(self, from_id: int):
        raise NotImplementedError("TODO")