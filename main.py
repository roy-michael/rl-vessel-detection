import os
import asyncio

import librosa
import matplotlib.pyplot as plt
import numpy as np

from agent import DispatcherAgent, SignalProcessorAgent
from environment import Environment


async def main():
    base_dir = "D:/RoyStudies/Recordings"
    base_dir = "D:/RoyStudies/Recordings/Garda_2_26/"
    base_dir = "D:/RoyStudies/Recordings/AUVExp_1_26/Itamar_AUV"
    croatia_base_dir = f"{base_dir}/Croatia/Ocean Sonics"
    # file_dir = f"{croatia_base_dir}/2507_1"
    # file_dir = f"{base_dir}/2_Deep Water/Electric"
    file_dir = f"{base_dir}/Part1_StraightLine/IcListen6695"    #IcListen6692"
    file_dir = f"{base_dir}/Part2_Poligon/IcListen6692"    #IcListen6692"
    min_freq = 100
    max_freq = 4000
    n_fft = 8 * 1024
    hop_length = n_fft // 16 # int(1 * 1024)
    window_sec = 15.0
    n_components = 8

    try:
        env = Environment(file_dir, min_freq, max_freq, n_fft, hop_length)
        agent = DispatcherAgent(env, min_freq, max_freq, n_fft, n_components, 2000, window_sec)
        signal_processor = SignalProcessorAgent(agent)

    except ValueError as e:
        print(e)
        return

    await env.start()
    
    # Wait for the environment to buffer the first frame so we have something to plot immediately
    first_frame, _ = await env.peek()
    if first_frame is None:
         print("No data available to start.")
         return
         
    # Start the agent, which will now handle the plotting and observation loop
    await agent.start()
    await signal_processor.start()
    
    # Keep the main thread alive while the agent runs
    # In a real UI application, this would be replaced by the UI event loop
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())