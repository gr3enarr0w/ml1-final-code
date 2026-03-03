import time
import random
import logging
import requests
# import gemini
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch
# import gym
import tqdm



class State():
    def __init__(self):
        self.state = "Initial State"

    def step(self, new_state):
        self.state = new_state
        print(f"State updated to: {self.state}")


class Actions:
    def __init__(self):
        pass

    def action_one(self):
        print("Action One executed.")

    def action_two(self):
        print("Action Two executed.")

    def action_three(self):
        print("Action Three executed.")

def main():

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting NCAA Prediction Market Simulation")





if __name__ == "__main__":

    main()
