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
  templates = np.array([np.roll(template_major, k) for k in range(0, 12)] + [np.roll(template_minor, k) for k in range(0, 12)])

  audiofilepath: str = "./" + str(sys.argv[1])

  y, sr = librosa.load(audiofilepath, sr=None, mono=True)
  y_harmonic, y_percussive = librosa.effects.hpss(y)

  onset_env = librosa.onset.onset_strength(y, sr=sr)
  tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

  playtime = int(y.size/sr)

  print(playtime)

  chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)

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
    chord_name = "C" if k == 0 else "C#" if k == 1 else "D" if k == 2 else "D#" if k == 3 else "E" if k == 4 else "F" if k == 5 else "F#" if k == 6 else "G" if k == 7 else "G#" if k == 8 else "A" if k == 9 else "A#" if k == 10 else "B" if k == 11 else "Cm" if k == 12 else "Cm#" if k == 13 else "Dm" if k == 14 else "Dm#" if k == 15 else "Em" if k == 16 else "Fm" if k == 17 else "Fm#" if k == 18 else "Gm" if k == 19 else "Gm#" if k == 20 else "Am" if k == 21 else "Am#" if k == 22 else "Bm"
    if k != before_chord_num:
      chord += chord_name + '\t'
    before_chord_num = k
  print(chord)

main()