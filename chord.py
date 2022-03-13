from logging import root
import numpy as np
import librosa
import librosa.display
import sys

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

  templates = np.array([np.roll(template_major, k) for k in range(0, 12)] + [np.roll(template_minor, k) for k in range(0, 12)] + \
    [np.roll(template_dim, k) for k in range(0, 12)] + [np.roll(template_aug, k) for k in range(0, 12)] + \
    [np.roll(template_sus4, k) for k in range(0, 12)] + [np.roll(template_sus2, k) for k in range(0, 12)] + \
    [np.roll(template_major_7, k) for k in range(0, 12)] + [np.roll(template_minor_7, k) for k in range(0, 12)] + \
    [np.roll(template_sus4_7, k) for k in range(0, 12)] + [np.roll(template_sus2_7, k) for k in range(0, 12)])

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
    chord_data[:, j] += chord_matching_score[:, k]
    i += 1
    if i >= delta_chroma:
      i = 0
      j += 1

  before_chord_num: int = -1
  chord: str = ""

  for i in range(0, len(chord_data[0])):
    k = np.argmax(chord_data[:,i])

    chord_name = "C" if k == 0 else "C#" if k == 1 else "D" if k == 2 else "D#" if k == 3 else "E" if k == 4 else "F" if k == 5 else "F#" if k == 6 else "G" if k == 7 else "G#" if k == 8 else "A" if k == 9 else "A#" if k == 10 else "B" if k == 11 else \
    "Cm" if k == 12 else "C#m" if k == 13 else "Dm" if k == 14 else "D#m" if k == 15 else "Em" if k == 16 else "Fm" if k == 17 else "F#m" if k == 18 else "Gm" if k == 19 else "G#m" if k == 20 else "Am" if k == 21 else "A#m" if k == 22 else "Bm" if k == 23 else \
    "Cdim" if k == 24 else "C#dim" if k == 25 else "Ddim" if k == 26 else "D#dim" if k == 27 else "Edim" if k == 28 else "Fdim" if k == 29 else "F#dim" if k == 30 else "Gdim" if k == 31 else "G#dim" if k == 32 else "Adim" if k == 33 else "A#dim" if k == 34 else "Bdim" if k == 35 else \
    "Caug" if k == 36 else "C#aug" if k == 37 else "Daug" if k == 38 else "D#aug" if k == 39 else "Eaug" if k == 40 else "Faug" if k == 41 else "F#aug" if k == 42 else "Gaug" if k == 43 else "G#aug" if k == 44 else "Aaug" if k == 45 else "A#aug" if k == 46 else "Baug" if k == 47 else \
    "Csus4" if k == 48 else "C#sus4" if k == 49 else "Dsus4" if k == 50 else "D#sus4" if k == 51 else "Esus4" if k == 52 else "Fsus4" if k == 53 else "F#sus4" if k == 54 else "Gsus4" if k == 55 else "G#sus4" if k == 56 else "Asus4" if k == 57 else "A#sus4" if k == 58 else "Bsus4" if k == 59 else \
    "Csus2" if k == 60 else "C#sus2" if k == 61 else "Dsus2" if k == 62 else "D#sus2" if k == 63 else "Esus2" if k == 64 else "Fsus2" if k == 65 else "F#sus2" if k == 66 else "Gsus2" if k == 67 else "G#sus2" if k == 68 else "Asus2" if k == 69 else "A#sus2" if k == 70 else "Bsus2" if k == 71 else \
    "C7" if k == 72 else "C#7" if k == 73 else "D7" if k == 74 else "D#7" if k == 75 else "E7" if k == 76 else "F7" if k == 77 else "F#7" if k == 78 else "G7" if k == 79 else "G#7" if k == 80 else "A7" if k == 81 else "A#7" if k == 82 else "B7" if k == 83 else \
    "Cm7" if k == 84 else "C#m7" if k == 85 else "Dm7" if k == 86 else "D#m7" if k == 87 else "Em7" if k == 88 else "Fm7" if k == 89 else "F#m7" if k == 90 else "Gm7" if k == 91 else "G#m7" if k == 92 else "Am7" if k == 93 else "A#m7" if k == 94 else "Bm7" if k == 95 else \
    "C7sus4" if k == 96 else "C#7sus4" if k == 97 else "D7sus4" if k == 98 else "D#7sus4" if k == 99 else "E7sus4" if k == 100 else "F7sus4" if k == 101 else "F#7sus4" if k == 102 else "G7sus4" if k == 103 else "G#7sus4" if k == 104 else "A7sus4" if k == 105 else "A#7sus4" if k == 106 else "B7sus4" if k == 107 else \
    "C7sus2" if k == 108 else "C#7sus2" if k == 109 else "D7sus2" if k == 110 else "D#7sus2" if k == 111 else "E7sus2" if k == 112 else "F7sus2" if k == 113 else "F#7sus2" if k == 114 else "G7sus2" if k == 115 else "G#7sus2" if k == 116 else "A7sus2" if k == 117 else "A#7sus2" if k == 118 else "B7sus2" #if k == 119 else \

    #if k != before_chord_num:
    chord += "{:12s}".format(chord_name)
    if (i + 1) % 16 == 0:
      chord += "\n"
    before_chord_num = k
  print(chord)

main()