# handpumpAquiferLstm
Use LSTM to model shallow aquifer level using handpump vibration data

scripts/LSTMmultiInUniOutWaterColExample.py is an example script to train/test a LSTM model to estimate the water column at a borehole using the frequency features generated from the vibration data collected at the handple of the handpump. The script uses modules and class defined in scripts/LSTMutils.py and scripts/LSTMmultiInUniOut.py.

data/dailyWaterColFreqFeatsPumpMP1.csv is an example dataset, which is used by scripts/LSTMmultiInUniOutWaterColExample.py.

Example outputs are shown in outputs/...

The example is tested using Python 3.7.3, sklearn 0.21.2, and keras 2.2.4.
