import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64


# from gtts import gTTS
# sound_file = BytesIO()
# tts = gTTS('Add text-to-speech to your app', lang='en')
# tts.write_to_fp(sound_file)



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


# def autoplay_audio(file_path: str):
#     with open(file_path, "rb") as f:
#         data = f.read()
#         b64 = base64.b64encode(data).decode()
#         md = f"""
#             <audio controls autoplay="true">
#             <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
#             </audio>
#             """
#         st.markdown(
#             md,
#             unsafe_allow_html=True,
#         )

## tts3 ##



st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Remove background from your image")
st.write(
    ":dog: Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:"
)
st.sidebar.write("## Upload and download :gear:")


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = remove(image)
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


## gTTS ##
# tell someingin here #
# st.audio(sound_file)

## tts3 ##
# autoplay_audio("./speech.mp3")
st.audio("speech.mp3")


if my_upload is not None:
    fix_image(upload=my_upload)
else:
    fix_image("./40469519171_fa183b8d38_z.jpg")
