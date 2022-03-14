import numpy as np
import librosa
import librosa.display
import sys

chord_dic = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B",
  12: "Cm", 13: "C#m", 14: "Dm", 15: "D#m", 16: "Em", 17: "Fm", 18: "F#m", 19: "Gm", 20: "G#m", 21: "Am", 22: "A#m", 23: "Bm",
  24: "Cdim", 25: "C#dim", 26: "Ddim", 27: "D#dim", 28: "Edim", 29: "Fdim", 30: "F#dim", 31: "Gdim", 32: "G#dim", 33: "Adim", 34: "A#dim", 35: "Bdim",
  36: "Caug", 37: "C#aug", 38: "Daug", 39: "D#aug", 40: "Eaug", 41: "Faug", 42: "F#aug", 43: "Gaug", 44: "G#aug", 45: "Aaug", 46: "A#aug", 47: "Baug",
  48: "Csus4", 49: "C#sus4", 50: "Dsus4", 51: "D#sus4", 52: "Esus4", 53: "Fsus4", 54: "F#sus4", 55: "Gsus4", 56: "G#sus4", 57: "Asus4", 58: "A#sus4", 59: "Bsus4",
  60: "Csus2", 61: "C#sus2", 62: "Dsus2", 63: "D#sus2", 64: "Esus2", 65: "Fsus2", 66: "F#sus2", 67: "Gsus2", 68: "G#sus2", 69: "Asus2", 70: "A#sus2", 71: "Bsus2",
  72: "C7", 73: "C#7", 74: "D7", 75: "D#7", 76: "E7", 77: "F7", 78: "F#7", 79: "G7", 80: "G#7", 81: "A7", 82: "A#7", 83: "B7",
  84: "Cm7", 85: "C#m7", 86: "Dm7", 87: "D#m7", 88: "Em7", 89: "Fm7", 90: "F#m7", 91: "Gm7", 92: "G#m7", 93: "Am7", 94: "A#m7", 95: "Bm7",
  96: "C7sus4", 97: "C#7sus4", 98: "D7sus4", 99: "D#7sus4", 100: "E7sus4", 101: "F7sus4", 102: "F#7sus4", 103: "G7sus4", 104: "G#7sus4", 105: "A7sus4", 106: "A#7sus4", 107: "B7sus4",
  108: "C7sus2", 109: "C#7sus2", 110: "D7sus2", 111: "D#7sus2", 112: "E7sus2", 113: "F7sus2", 114: "F#7sus2", 115: "G7sus2", 116: "G#7sus2", 117: "A7sus2", 118: "A#7sus2", 119: "B7sus2",
  120: "Cdim7", 121: "C#dim7", 122: "Ddim7", 123: "D#dim7", 124: "Edim7", 125: "Fdim7", 126: "F#dim7", 127: "Gdim7", 128: "G#dim7", 129: "Adim7", 130: "A#dim7", 131: "Bdim7",
  132: "Caug7", 133: "C#aug7", 134: "Daug7", 135: "D#aug7", 136: "Eaug7", 137: "Faug7", 138: "F#aug7", 139: "Gaug7", 140: "G#aug7", 141: "Aaug7", 142: "A#aug7", 143: "Baug7",
  144: "C5", 145: "C#5", 146: "D5", 147: "D#5", 148: "E5", 149: "F5", 150: "F#5", 151: "G5", 152: "G#5", 153: "A5", 154: "A#5", 155: "B5",
}

def main():
  key_shift: int = 0
  if len(sys.argv) == 3:
    key_shift = -(int)(sys.argv[2])

  elif len(sys.argv) != 2:
    print("ファイルを一つ指定してください")
    return
  
  root = 1.0
  third = 0.5
  fifth = -1.5
  seventh = 0.0

  #chord created with 2 tones
  base2 = 6.0
  root_tone2 = base2
  third_tone2 = -9.0
  fifth_tone2 = base2

  #chord created with 3 tones
  base3 = 4.0
  root_tone3 = base3 + root
  third_tone3 = base3 + third
  fifth_tone3 = base3 + fifth

  #chord created with 4 tones
  base4 = 3.5
  root_tone4 = base4 + (root * 0.875)
  third_tone4 = base4 + (third * 0.875)
  fifth_tone4 = base4 + (fifth * 0.875)
  seventh_tone4 = base4 + (seventh * 0.875)

  #3tones chord
  template_major = np.roll(np.array([root_tone3, 0.0, 0.0, 0.0, third_tone3, 0.0, 0.0, fifth_tone3, 0.0, 0.0, 0.0, 0.0]), key_shift)
  template_minor = np.roll(np.array([root_tone3, 0.0, 0.0, third_tone3, 0.0, 0.0, 0.0, fifth_tone3, 0.0, 0.0, 0.0, 0.0]), key_shift)
  template_dim = np.roll(np.array([root_tone3, 0.0, 0.0, third_tone3, 0.0, 0.0, fifth_tone3, 0.0, 0.0, 0.0, 0.0, 0.0]), key_shift)
  template_aug = np.roll(np.array([root_tone3, 0.0, 0.0, 0.0, third_tone3, 0.0, 0.0, 0.0, fifth_tone3, 0.0, 0.0, 0.0]), key_shift)
  template_sus4 = np.roll(np.array([root_tone3, 0.0, 0.0, 0.0, 0.0, third_tone3, 0.0, fifth_tone3, 0.0, 0.0, 0.0, 0.0]), key_shift)
  template_sus2 = np.roll(np.array([root_tone3, 0.0, third_tone3, 0.0, 0.0, 0.0, 0.0, fifth_tone3, 0.0, 0.0, 0.0, 0.0]), key_shift)

  #4tones chord
  template_major_7 = np.roll(np.array([root_tone4, 0.0, 0.0, 0.0, third_tone4, 0.0, 0.0, fifth_tone4, 0.0, 0.0, seventh_tone4, 0.0]), key_shift)
  template_minor_7 = np.roll(np.array([root_tone4, 0.0, 0.0, third_tone4, 0.0, 0.0, 0.0, fifth_tone4, 0.0, 0.0, seventh_tone4, 0.0]), key_shift)
  template_sus4_7 = np.roll(np.array([root_tone4, 0.0, 0.0, 0.0, 0.0, third_tone4, 0.0, fifth_tone4, 0.0, 0.0, seventh_tone4, 0.0]), key_shift)
  template_sus2_7 = np.roll(np.array([root_tone4, 0.0, third_tone4, 0.0, 0.0, 0.0, 0.0, fifth_tone4, 0.0, 0.0, seventh_tone4, 0.0]), key_shift)
  template_dim_7 = np.roll(np.array([root_tone4, 0.0, 0.0, third_tone4, 0.0, 0.0, fifth_tone4, 0.0, 0.0, 0.0, 0.0, 0.0]), key_shift)
  template_aug_7 = np.roll(np.array([root_tone4, 0.0, 0.0, 0.0, third_tone4, 0.0, 0.0, 0.0, fifth_tone4, 0.0, 0.0, 0.0]), key_shift)

  #power chord
  template_power = np.roll(np.array([root_tone2, 0.0, 0.0, 0.0, third_tone2, 0.0, 0.0, fifth_tone2, 0.0, 0.0, 0.0, 0.0]), key_shift)

  templates = np.array([np.roll(template_major, k) for k in range(0, 12)] + [np.roll(template_minor, k) for k in range(0, 12)] + 
    [np.roll(template_dim, k) for k in range(0, 12)] + [np.roll(template_aug, k) for k in range(0, 12)] + 
    [np.roll(template_sus4, k) for k in range(0, 12)] + [np.roll(template_sus2, k) for k in range(0, 12)] + 
    [np.roll(template_major_7, k) for k in range(0, 12)] + [np.roll(template_minor_7, k) for k in range(0, 12)] + 
    [np.roll(template_sus4_7, k) for k in range(0, 12)] + [np.roll(template_sus2_7, k) for k in range(0, 12)] + 
    [np.roll(template_dim_7, k) for k in range(0, 12)] + [np.roll(template_aug_7, k) for k in range(0, 12)] + 
    [np.roll(template_power, k) for k in range(0,12)]
  )

  audiofilepath: str = f"./{sys.argv[1]}"

  yt, sr = librosa.load(audiofilepath, sr=None, mono=True)
  y, _sr = librosa.effects.trim(yt)
  y_harmonic, y_percussive = librosa.effects.hpss(y)

  onset_env = librosa.onset.onset_strength(y, sr=sr)
  tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

  playtime = int(y.size/sr)

  print(f"再生時間:{playtime}")
  print(f"テンポ:{tempo}")

  chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

  delta_chroma = int(2*60*len(chroma[0])/(tempo*playtime))

  chord_matching_score = np.dot(templates, chroma) #templatesとchromaの内積
  chord_data = np.zeros((len(chord_matching_score), int(len(chord_matching_score[0])/delta_chroma + 1)))

  i: int = 0
  j: int = 0
  for k in range(0, len(chord_matching_score[0])):
    if (i > (delta_chroma * 0.2) and i < (delta_chroma * 0.8)):
      chord_data[:, j] += chord_matching_score[:, k]
    i += 1
    if i >= delta_chroma:
      i = 0
      j += 1

  chord: str = ""

  for i in range(0, len(chord_data[0])):
    k = np.argmax(chord_data[:,i])

    chord_name = chord_dic[k]
    chord += "{:12s}".format(chord_name)
    if (i + 1) % 16 == 0:
      chord += "\n"

  print(chord)

main()