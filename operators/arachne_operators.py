from .arachne import localisation 
from keras.engine.sequential import Sequential
from numpy import ndarray

def operator_arachne(model: Sequential, i_pos: ndarray = [], i_neg: ndarray = []) -> None:
    assert model is not None, "Model not found"
    assert len(i_pos) > 0, "No positive examples"
    assert len(i_neg) > 0, "No negative examples"
    assert len(i_neg) >= len(i_pos), "More positive examples than negative examples"

    localisation.bidirectional_localisation(model, i_pos, i_neg)
