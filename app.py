import streamlit as st
st.set_page_config(
    page_title="Music Generation",
    page_icon=":musical_score"
)
import io
import numpy as np
from eval import generate_song_notes, EMOTION_LABEL_TO_NUMBER
from scipy.io import wavfile
from amc_dl.format_convert import get_midi_from_notes

st.write(
    """
    Generating songs with emotions
    """
)

def emuseBERT_generation():
    st.title(":musical_note: Generate emotional music using eMuseBERT")
    selected_emotion = st.selectbox("What emotion do you expect?", ("Q1", "Q2", "Q3", "Q4"))
    num_steps = st.slider("Number of Markov Chain steps", 0, 10000, 4000, 50)
    num_corrupt_steps = st.slider("Number of Markov Chain corruption steps", 0, num_steps, int(num_steps/4), 50)

    if st.button("Generate"):
        with st.spinner("Generating in progress. This might take several minutes."):
            emotion_number = EMOTION_LABEL_TO_NUMBER[selected_emotion]

            midi_notes = generate_song_notes(emotion_number, num_steps, num_corrupt_steps)
            music = get_midi_from_notes(midi_notes)
            audio_data = music.fluidsynth()
            audio_data = np.int16(
                audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
            )
            virtual_file = io.BytesIO()
            wavfile.write(virtual_file, 44100, audio_data)
            st.audio(virtual_file)
            st.markdown("Download the audio by right-clicking on the media player")


if __name__ == "__main__":
    emuseBERT_generation()



