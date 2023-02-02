from .arachne import localisation 

def operator_arachne(model, i_pos = [], i_neg = []):
    assert model is not None, "Model not found"
    assert len(i_pos) > 0, "No positive examples"
    assert len(i_neg) > 0, "No negative examples"
    assert len(i_neg) >= len(i_pos), "More positive examples than negative examples"

    localisation.bidirectional_localisation(model, i_pos, i_neg)
