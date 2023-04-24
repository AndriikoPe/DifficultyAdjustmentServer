import torch
from flask import Flask, request, jsonify, Response
from dataclasses import dataclass


@dataclass
class DataEntry:
    health: float
    healthToTime: float
    timeElapsed: float
    damagedLastWave: float
    avgWaveDamage: float
    factorDifference: float
    currentDifficulty: float


app = Flask(__name__)
model = torch.load('trained_model_1.pt')


@app.route('/', methods=['POST'])
def predict() -> Response:
    data = request.json['data']
    data_entry = DataEntry(**data)
    input_data = torch.tensor([data_entry.health,
                               data_entry.healthToTime,
                               data_entry.timeElapsed / 15116.16,
                               data_entry.damagedLastWave,
                               data_entry.avgWaveDamage,
                               data_entry.factorDifference,
                               data_entry.currentDifficulty / 2.0]).reshape(1, 7)

    output1, output2 = model(input_data)
    action = output2.detach().numpy()[0][0]

    return jsonify({'action': float(action)})


if __name__ == '__main__':
    app.run(debug=True)
