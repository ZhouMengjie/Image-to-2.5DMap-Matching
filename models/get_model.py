from models.seq_model import TransMixer

def get_model(model_type, no_share):
    if model_type == 'transmixer' and no_share:
        model_pano = TransMixer()
