# 🃏 Reinforcement Learning Poker Bot using MCCFR

This repository contains an implementation of a **Heads Up No Limit (HUNL) Poker Bot** trained using **Monte Carlo Counterfactual Regret Minimization (MCCFR)** — a state-of-the-art reinforcement learning algorithm for solving large imperfect-information games like Poker.

---

## 🚀 Overview

The goal of this project is to train an AI agent capable of playing heads-up No-Limit Texas Hold’em (or other variants) competitively using self-play and regret minimization techniques. Unlike traditional tabular approaches, MCCFR uses sampled trajectories to scale to larger game trees and efficiently converge toward Nash equilibrium strategies.

---

## 🧠 Key Features

- **MCCFR Algorithm**: Implements external sampling MCCFR for scalable learning.
- **Game Engine**: Using Openspiels Poker enviroment with diffrent sets of abstractions.
- **Strategy Abstraction**: Hand strength bucketing and action abstraction for tractable play.
- **Modular Design**: Easy to swap out game variants, abstractions, and sampling policies.

---

## 🧱 Project Structure

**TBA**

---

## 📈 How It Works

1. **Information Set Generation**: The game state is abstracted into an information set that includes hidden and visible information (e.g., private cards, public board, betting history).
2. **Regret Sampling**: The agent samples trajectories and updates regrets for actions based on counterfactual values.
3. **Strategy Averaging**: Over iterations, the average strategy (rather than the current one) is recorded as the output.
4. **Convergence**: With enough iterations, the strategy converges to a Nash equilibrium.

---

## 📚 Resources
[Superhuman AI for multiplayer poker (Brown Et Al.)](https://www.science.org/doi/10.1126/science.aay2400)

[OpenSpiel](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/mccfr.py)
