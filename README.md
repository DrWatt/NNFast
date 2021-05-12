# NNFast

Script for quick training and testing of Neural Networks using Keras.

Example:
```bash
python NNfast.py --data out_data.csv  --nnparams 10 0 0 --nnlayout lay.json --targetvar genParticle.pt
```
Training of a Densely Connected NN for 10 epochs with data from `out_data.csv`, target variable `genParticle.pt` and layout described in `lay.json`.
