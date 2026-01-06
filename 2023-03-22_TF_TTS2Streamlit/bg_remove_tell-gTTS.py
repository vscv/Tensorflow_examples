# -*- coding: utf-8 -*-
import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64

# +
## tts3 ##
import pyttsx3
# 初始化
engine = pyttsx3.init()

# 獲取語音包
voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[0].id)    #changing index, changes voices. o for male # not works!

for voice in voices:
#     print('id = {} \tname = {} \n'.format(voice.id, voice.name))
   print(voice)

# 設置語音包
engine.setProperty('voice', 'zh') #'english') #'Mandarin')
# engine.setProperty('voice', voices[0].id)    #changing index, changes voices. o for male # not works!
# engine.setProperty('voice', 'Mandarin') #'english') #'Mandarin')

# 語速控制
rate = engine.getProperty('rate')
print('語速', rate)
engine.setProperty('rate', 150)

# 音量控制
volume = engine.getProperty('volume')
print("音量", volume)
# engine.setProperty('volume', volume-0.25)

string = "森山小姐，我相信你。"
# string = '''这段代码是官方给出'''
# string = "command not found command not found"
#engine.say(string) #aplay: command not found [aplay is a command-line utility for playing audio files on a Linux system. ]
engine.save_to_file(string, 'speech.mp3')
engine.runAndWait()


# -

# # tts3 ##

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
         <audio controls autoplay="true">
         <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
         </audio>
         """
        st.markdown(
         md,
         unsafe_allow_html=True,
     )

# # gTTS # #
from gtts import gTTS
sound_file = BytesIO()
tts = gTTS('森山小姐，我相信你。Add text-to-speech to your app', lang='zh-TW')
tts.write_to_fp(sound_file)
tts.save("speech.mp3")
# # gTTS # #



# # Stresmlit web app # #
st.set_page_config(layout="wide", page_title="OCC to gTTS")
st.write("## OCR image and Speak it out")
st.write(
    "📄🔊 Try uploading an image to auto-speak the text. "
)
st.sidebar.write("## Upload and download :gear:")
# # Stresmlit web app # #


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image 📄")
    col1.image(image)

    fixed = remove(image)
    col2.write("OCR to Text 💬")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download OCR image", convert_image(fixed), "ocred.png", "image/png")
#    st.write("# Auto-playing Audio!")
#    autoplay_audio("speech.mp3")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


# # gTTS # #
# tell someingin here #
st.write("## 🔊 Auto Speaking the OCR  🔊 ")
autoplay_audio("speech.mp3")
#st.audio(sound_file)


# # tts3 # #
# autoplay_audio("./speech.mp3")
# st.audio("speech.mp3")


if my_upload is not None:
    fix_image(upload=my_upload)
else:
    fix_image("./40469519171_fa183b8d38_z.jpg")
