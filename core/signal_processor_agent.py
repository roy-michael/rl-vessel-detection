import numpy as np
from core.vessel_state import VesselState

class SignalProcessorAgent:
    """
    A child agent of DispatcherAgent.
    Represents and manages the tracking lifecycle of a single physical vessel.
    Uses the DispatcherAgent as its environment.
    """
    def __init__(self, env, start_time, initial_freq, initial_spread, initial_amp, vessel_id):
        self.env = env  # The DispatcherAgent is the environment
        self.vessel_id = vessel_id
        self.vessel_state = VesselState(start_time, initial_freq, initial_spread, initial_amp, vessel_id)
        self.last_seen_time = start_time
        self.is_active = True

    def add_observation(self, freq, spread, amp, current_time):
        self.vessel_state.add_observation(freq, spread, amp)
        self.last_seen_time = current_time

    def terminate(self, end_time):
        self.is_active = False
        self.vessel_state.close(end_time)
