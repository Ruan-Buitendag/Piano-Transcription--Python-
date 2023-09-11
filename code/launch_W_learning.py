import learning_functions as scr

if __name__ == "__main__":
    path_maps = "../MAPS"

    pianos = ["AkPnBcht", "AkPnBsdf", "AkPnStgb", "ENSTDkAm", "SptkBGAm", "StbgTGd2", "AkPnCGdD", "ENSTDkCl"]

    piano = "AkPnBcht"

    type = "stft"
    # type = "mspec"

    num_bins = 4096

    note_intensity = "M"
    itmax = 500
    path_piano_isol = "{}/{}/ISOL/NO/".format(path_maps, piano)

    beta = 1

    T = 10
    _, _ = scr.learning_W_and_persist(path_piano_isol, beta, T, itmax=itmax, rank=1, init="L1", piano_type=piano,
                                      note_intensity=note_intensity, spec_type=type, num_points=num_bins)

    print("Done")
