# DifficultyAdjustmentServer

This is a python server for the [DifficultyAdjustment](https://github.com/MrPelmen/DifficultyAdjustment) iOS project. Follow the instructions below to get started.

## Getting Started

### Prerequisites

You'll need the following installed on your local machine:

- Python 3.x

### Installation

1. Clone this repository to your local machine.
2. Start the server by running `main.py`.
3. The server will expose an endpoint on `localhost:5000`.

### Usage

1. Send an HTTP 'POST' request to `localhost:5000` with the following payload:

    ```
    {
        "data": {
            "health": float,
            "healthToTime": float,
            "timeElapsed": float,
            "damagedLastWave": float,
            "avgWaveDamage": float,
            "factorDifference": float,
            "currentDifficulty": float
        }
    }
    ```

2. You should receive a response like this:

    ```
    {
        "action": float
    }
    ```
