# The Acoustic Counterfeit Machine
#### A portion of:
### Prying Ears: A Critical Response to Mass Audio Surveillance
#### A Master's Thesis
#### 2018-2019

Full Thesis can be found here: https://aaronmkarp.com/files/KarpThesis.pdf


#### Overview

Named the Acoustic Counterfeit Machine, or ACM, this system is designed to hide speech from methods of mass audio surveillance, and
to do so in such a way as to not arouse suspicion. The ACM is built in the following structure: 

  1. A comparative database is created to calculate approximate matches between the spectral features of spoken words
  2. That database is used to generate training data to an LSTM (Long Short Term Memory) neural network, designed to match spectrally similar audio in a real-time context
  3. Live input is fed through the neural network and an ideal mask is calculated
  4. The calculate mask is played over a speaker, with a delay of <15ms between the reception of the input audio and the sonification of the matching mask


![SystemDiagram](https://github.com/aaronkarp123/thesis/blob/master/diagrams/UpdatedSystem.png)
