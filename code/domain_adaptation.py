import single_note_eq_mask as eq
import numpy as np


def EQDictionary(dictionary, piano_H, piano_W):
    # dict_W = np.copy(dictionary)
    dict_W = np.zeros(dictionary.shape)

    for note in range(88):
        eq_mask = eq.CalculateEQMask(str(note + 21), piano_H, piano_W)

        template_eq = eq.TemplateEQ(eq_mask)

        aaa = (template_eq.T * 0.025)

        dict_W[:, :, note] = dictionary[:, :, note] + (template_eq.T * 0.025)
        dict_W[:, :, note][dict_W[:, :, note] < 0] = 0.0000000001
        # template_eq = eq.TemplateEQWithTime(scaled_eq)
        # balls[:, :, note] = dict_W[:, :, note] + (template_eq.T * 0.05)

    # dict_W = dictionary
    return dict_W




def EQDictionaryFromSingleNote(dictionary, piano_H, piano_W, midi_note_for_eq):
    dict_W = np.zeros(dictionary.shape)

    for note in range(88):
        eq_mask = eq.CalculateEQMask(str(note + 21), piano_H, piano_W)

        scaled_eq = eq.ScaleSpectrogramNoStretch(eq_mask, midi_note_for_eq, note + 21)

        template_eq = eq.TemplateEQ(scaled_eq)
        dict_W[:, :, note] = dict_W[:, :, note] + (template_eq.T * 0.025)
        dict_W[:, :, note][dict_W[:, :, note] < 0] = 0.0000000001
        # template_eq = eq.TemplateEQWithTime(scaled_eq)
        # balls[:, :, note] = dict_W[:, :, note] + (template_eq.T * 0.05)

    # dict_W = dictionary
    return dict_W
