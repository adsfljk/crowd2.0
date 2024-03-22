import sys
import io, os, stat
import subprocess
import random
from zipfile import ZipFile
import uuid
import time
import torch
import torchaudio


#download for mecab
# os.system('python -m unidic download')

import base64
import csv
from io import StringIO
import datetime
import re

import gradio as gr
from scipy.io.wavfile import write
from pydub import AudioSegment

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir

model_path = "checkpoint"
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir=model_path,
    checkpoint_path=os.path.join(model_path, "model.pth"),
    vocab_path=os.path.join(model_path, "vocab.json"),
    eval=True,
    use_deepspeed=False,
)
model.cuda()

supported_languages = config.languages

def predict(
    prompt,
    language,
    audio_file_pth,
    mic_file_path,
    use_mic,
):
    if language not in supported_languages:
        gr.Warning(
            f"Language you put {language} in is not in is not in our Supported Languages, please choose from dropdown"
        )

        return (
            None,
            None,
            None,
            None,
        )

    # language_predicted = langid.classify(prompt)[
    #     0
    # ].strip()  # strip need as there is space at end!

    # # tts expects chinese as zh-cn
    # if language_predicted == "zh":
    #     # we use zh-cn
    #     language_predicted = "zh-cn"

    # print(f"Detected language:{language_predicted}, Chosen language:{language}")

    # # After text character length 15 trigger language detection
    # if len(prompt) > 15:
    #     # allow any language for short text as some may be common
    #     # If user unchecks language autodetection it will not trigger
    #     # You may remove this completely for own use
    #     if language_predicted != language and not no_lang_auto_detect:
    #         # Please duplicate and remove this check if you really want this
    #         # Or auto-detector fails to identify language (which it can on pretty short text or mixed text)
    #         gr.Warning(
    #             f"It looks like your text isn’t the language you chose , if you’re sure the text is the same language you chose, please check disable language auto-detection checkbox"
    #         )

    #         return (
    #             None,
    #             None,
    #             None,
    #             None,
    #         )

    if use_mic == True:
        if mic_file_path is not None:
            speaker_wav = mic_file_path
        else:
            gr.Warning(
                "Please record your voice with Microphone, or uncheck Use Microphone to use reference audios"
            )
            return (
                None,
                None,
                None,
                None,
            )

    else:
        speaker_wav = audio_file_pth

    if len(prompt) < 2:
        gr.Warning("Please give a longer prompt text")
        return (
            None,
            None,
            None,
            None,
        )
    if len(prompt) > 200:
        gr.Warning(
            "Text length limited to 200 characters for this demo, please try shorter text. You can clone this space and edit code for your own usage"
        )
        return (
            None,
            None,
            None,
            None,
        )

    metrics_text = ""
    t_latent = time.time()

    # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
    try:
        (
            gpt_cond_latent,
            speaker_embedding,
        ) = model.get_conditioning_latents(audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)
    except Exception as e:
        print("Speaker encoding error", str(e))
        gr.Warning(
            "It appears something wrong with reference, did you unmute your microphone?"
        )
        return (
            None,
            None,
            None,
            None,
        )

    latent_calculation_time = time.time() - t_latent
    # metrics_text=f"Embedding calculation time: {latent_calculation_time:.2f} seconds\n"

    # temporary comma fix
    prompt= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2",prompt)

    wav_chunks = []
    ## Direct mode
    
    print("I: Generating new audio...")
    t0 = time.time()
    out = model.inference(
        prompt,
        language,
        gpt_cond_latent,
        speaker_embedding,
        repetition_penalty=5.0,
        temperature=0.75,
    )
    inference_time = time.time() - t0
    print(f"I: Time to generate audio: {round(inference_time*1000)} milliseconds")
    metrics_text+=f"Time to generate audio: {round(inference_time*1000)} milliseconds\n"
    real_time_factor= (time.time() - t0) / out['wav'].shape[-1] * 24000
    print(f"Real-time factor (RTF): {real_time_factor}")
    metrics_text+=f"Real-time factor (RTF): {real_time_factor:.2f}\n"
    torchaudio.save("output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)


    """
    print("I: Generating new audio in streaming mode...")
    t0 = time.time()
    chunks = model.inference_stream(
        prompt,
        language,
        gpt_cond_latent,
        speaker_embedding,
        repetition_penalty=7.0,
        temperature=0.85,
    )

    first_chunk = True
    for i, chunk in enumerate(chunks):
        if first_chunk:
            first_chunk_time = time.time() - t0
            metrics_text += f"Latency to first audio chunk: {round(first_chunk_time*1000)} milliseconds\n"
            first_chunk = False
        wav_chunks.append(chunk)
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
    inference_time = time.time() - t0
    print(
        f"I: Time to generate audio: {round(inference_time*1000)} milliseconds"
    )
    #metrics_text += (
    #    f"Time to generate audio: {round(inference_time*1000)} milliseconds\n"
    #)

    wav = torch.cat(wav_chunks, dim=0)
    print(wav.shape)
    real_time_factor = (time.time() - t0) / wav.shape[0] * 24000
    print(f"Real-time factor (RTF): {real_time_factor}")
    metrics_text += f"Real-time factor (RTF): {real_time_factor:.2f}\n"

    torchaudio.save("output.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
    """

    return (
        "output.wav",
        metrics_text,
        speaker_wav,
    )


title = "Coqui🐸 XTTS"

description = """

<br/>

This demo is currently running **XTTS v2.0.3** <a href="https://huggingface.co/coqui/XTTS-v2">XTTS</a> is a multilingual text-to-speech and voice-cloning model. This demo features zero-shot voice cloning, however, you can fine-tune XTTS for better results. Leave a star 🌟 on Github <a href="https://github.com/coqui-ai/TTS">🐸TTS</a>, where our open-source inference and training code lives.

<br/>

Supported languages: Arabic: ar, Brazilian Portuguese: pt , Mandarin Chinese: zh-cn, Czech: cs, Dutch: nl, English: en, French: fr, German: de, Italian: it, Polish: pl, Russian: ru, Spanish: es, Turkish: tr, Japanese: ja, Korean: ko, Hungarian: hu, Hindi: hi

<br/>
"""

links = """
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=0d00920c-8cc9-4bf3-90f2-a615797e5f59" />

|                                 |                                         |
| ------------------------------- | --------------------------------------- |
| 🐸💬 **CoquiTTS**                | <a style="display:inline-block" href='https://github.com/coqui-ai/TTS'><img src='https://img.shields.io/github/stars/coqui-ai/TTS?style=social' /></a>|
| 💼 **Documentation**            | [ReadTheDocs](https://tts.readthedocs.io/en/latest/)
| 👩‍💻 **Questions**                | [GitHub Discussions](https://github.com/coqui-ai/TTS/discussions) |
| 🗯 **Community**         | [![Dicord](https://img.shields.io/discord/1037326658807533628?color=%239B59B6&label=chat%20on%20discord)](https://discord.gg/5eXr5seRrv)  |


"""

article = """
<div style='margin:20px auto;'>
<p>By using this demo you agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml</p>
<p>We collect data only for error cases for improvement.</p>
</div>
"""
examples = [
    [
        "Once when I was six years old I saw a magnificent picture",
        "en",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Lorsque j'avais six ans j'ai vu, une fois, une magnifique image",
        "fr",
        "examples/male.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Als ich sechs war, sah ich einmal ein wunderbares Bild",
        "de",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Cuando tenía seis años, vi una vez una imagen magnífica",
        "es",
        "examples/male.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Quando eu tinha seis anos eu vi, uma vez, uma imagem magnífica",
        "pt",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Kiedy miałem sześć lat, zobaczyłem pewnego razu wspaniały obrazek",
        "pl",
        "examples/male.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Un tempo lontano, quando avevo sei anni, vidi un magnifico disegno",
        "it",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Bir zamanlar, altı yaşındayken, muhteşem bir resim gördüm",
        "tr",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Когда мне было шесть лет, я увидел однажды удивительную картинку",
        "ru",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Toen ik een jaar of zes was, zag ik op een keer een prachtige plaat",
        "nl",
        "examples/male.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Když mi bylo šest let, viděl jsem jednou nádherný obrázek",
        "cs",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "当我还只有六岁的时候， 看到了一副精彩的插画",
        "zh-cn",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "かつて 六歳のとき、素晴らしい絵を見ました",
        "ja",
        "examples/female.wav",
        None,
        False,
        True,
        False,
        True,
    ],
    [
        "한번은 내가 여섯 살이었을 때 멋진 그림을 보았습니다.",
        "ko",
        "examples/female.wav",
        None,
        False,
        True,
        False,
        True,
    ],
        [
        "Egyszer hat éves koromban láttam egy csodálatos képet",
        "hu",
        "examples/male.wav",
        None,
        False,
        True,
        False,
        True,
    ],
]



with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ## <img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/coqui-log-green-TTS.png" height="56"/>
                """
            )
        with gr.Column():
            # placeholder to align the image
            pass

    with gr.Row():
        with gr.Column():
            gr.Markdown(description)
        with gr.Column():
            gr.Markdown(links)

    with gr.Row():
        with gr.Column():
            input_text_gr = gr.Textbox(
                label="Text Prompt",
                info="One or two sentences at a time is better. Up to 200 text characters.",
                value="Hi there, I'm your new voice clone. Try your best to upload quality audio.",
            )
            language_gr = gr.Dropdown(
                label="Language",
                info="Select an output language for the synthesised speech",
                choices=[
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "pl",
                    "tr",
                    "ru",
                    "nl",
                    "cs",
                    "ar",
                    "zh-cn",
                    "ja",
                    "ko",
                    "hu",
                    "hi"
                ],
                max_choices=1,
                value="en",
            )
            ref_gr = gr.Audio(
                label="Reference Audio",
                # info="Click on the ✎ button to upload your own target speaker audio",
                type="filepath",
                value="examples/female.wav",
            )
            mic_gr = gr.Audio(
                sources="microphone",
                type="filepath",
                # info="Use your microphone to record audio",
                label="Use Microphone for Reference",
            )
            use_mic_gr = gr.Checkbox(
                label="Use Microphone",
                value=False,
                info="Notice: Microphone input may not work properly under traffic",
            )

            tts_button = gr.Button("Send", elem_id="send-btn", visible=True)


        with gr.Column():
            audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
            out_text_gr = gr.Text(label="Metrics")
            ref_audio_gr = gr.Audio(label="Reference Audio Used")

    with gr.Row():
        gr.Examples(examples,
                    label="Examples",
                    inputs=[input_text_gr, language_gr, ref_gr, mic_gr, use_mic_gr],
                    outputs=[audio_gr, out_text_gr, ref_audio_gr],
                    fn=predict,
                    cache_examples=False,)

    tts_button.click(predict, [input_text_gr, language_gr, ref_gr, mic_gr, use_mic_gr], outputs=[audio_gr, out_text_gr, ref_audio_gr])

demo.queue()  
demo.launch(server_name="0.0.0.0",server_port=11451,debug=True, show_api=True)