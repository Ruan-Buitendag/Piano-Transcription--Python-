import single_note_eq_mask as eq
import numpy as np


def EQDictionary(dictionary, piano_H, piano_W):
    # dict_W = np.copy(dictionary)
    dict_W = np.zeros(dictionary.shape)

    for note in range(88):
        eq_mask = eq.CalculateEQMask(str(note + 21), piano_H, piano_W, dictionary)

        template_eq = eq.TemplateEQ(eq_mask)

        dict_W[:, :, note] = dictionary[:, :, note] * template_eq


    # dict_W = dictionary
    return dict_W



def EQDictionaryFromSingleNote(dictionary, piano_H, piano_W, midi_note_for_eq):
    dict_W = np.zeros(dictionary.shape)

    for note in range(88):
        eq_mask = eq.CalculateEQMask(str(note + 21), piano_H, piano_W, dictionary)

        eq_mask[eq_mask == 1] = 0

        scaled_eq = eq.ScaleSpectrogramNoStretch(eq_mask, midi_note_for_eq, note + 21)

        scaled_eq[scaled_eq == 0] = 1

        template_eq = eq.TemplateEQ(scaled_eq)

        dict_W[:, :, note] = dictionary[:, :, note] * template_eq

    # dict_W = dictionary
    return dict_W
