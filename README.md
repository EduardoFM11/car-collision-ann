# car-collision-ann

This project consists in the training of a neural network that receives a dataset
of vehicle and driver properties to try predict whether a collision is about to
happen with an unexpected object.

The dataset properties are moments before encountering the object:

- Speed [0-120km/h]
- Terrain quality [0-10]
- Vision angle [0-360º]
- Driver experience [0-400,000km]

Since the raw data is not understandable by the ANN, the data need to be scaled with
a min-max range, in this case from 0 to 1.
  
