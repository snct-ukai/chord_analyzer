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
  
  template_major = np.roll(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), key_shift)
  template_minor = np.roll(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), key_shift)
  template_dim = np.roll(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), key_shift)
  template_aug = np.roll(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), key_shift)
  templates = np.array([np.roll(template_major, k) for k in range(0, 12)] + [np.roll(template_minor, k) for k in range(0, 12)] + [np.roll(template_dim, k) for k in range(0, 12)] + [np.roll(template_aug, k) for k in range(0, 12)])

  audiofilepath: str = f"./{sys.argv[1]}"

  yt, sr = librosa.load(audiofilepath, sr=None, mono=True)
  y, _sr = librosa.effects.trim(yt)

  onset_env = librosa.onset.onset_strength(y, sr=sr)
  tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

  playtime = int(y.size/sr)

  print(f"再生時間:{playtime}")
  print(f"テンポ:{tempo}")

  chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

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
    "Cm" if k == 12 else "Cm#" if k == 13 else "Dm" if k == 14 else "Dm#" if k == 15 else "Em" if k == 16 else "Fm" if k == 17 else "Fm#" if k == 18 else "Gm" if k == 19 else "Gm#" if k == 20 else "Am" if k == 21 else "Am#" if k == 22 else "Bm" if k == 23 else \
    "Cdim" if k == 24 else "C#dim" if k == 25 else "Ddim" if k == 26 else "D#dim" if k == 27 else "Edim" if k == 28 else "Fdim" if k == 29 else "F#dim" if k == 30 else "Gdim" if k == 31 else "G#dim" if k == 32 else "Adim" if k == 33 else "A#dim" if k == 34 else "Bdim" if k == 35 else \
    "C" if k == 36 else "C#" if k == 37 else "D" if k == 38 else "D#" if k == 39 else "E" if k == 40 else "F" if k == 41 else "F#" if k == 42 else "G" if k == 43 else "G#" if k == 44 else "A" if k == 45 else "A#" if k == 46 else "B"# if k == 47 else

    if k != before_chord_num:
      chord += chord_name + '\t'
    before_chord_num = k
  print(chord)

main()