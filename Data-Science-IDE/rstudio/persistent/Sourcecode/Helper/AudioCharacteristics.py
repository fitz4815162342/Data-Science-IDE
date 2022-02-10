class AudioCharacteristics:
    """Class to encapsulate the characteristics of an audio sample"""
    
    def __init__(self, sample_name, time_series, sampling_rate):
        self.sample_name = sample_name
        self.time_series = time_series
        self.sampling_rate = sampling_rate
        self.features = {
            "MFCC": None,
            "ZeroCrossingRate": None,
            "Energy": None,
            "SpectralRollOff": None,
            "SpectralFlux": None,
            "ChromaFeatures": None,
            "Pitch": None
        }
