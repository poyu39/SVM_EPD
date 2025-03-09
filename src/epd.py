import argparse
import gradio as gr
import torch
import torchaudio
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path
from model import SVM_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_speech_segments(waveform, sr, frame_size=400, hop_size=160, n_mfcc=13):
    """
    偵測語音端點，將 wav 切割成連續的 speech frame。
    
    Args:
        waveform (torch.Tensor):    音訊波形
        sr (int):                   取樣率
        frame_size (int):           每個 frame 的大小
        hop_size (int):             frame 的移動步長
        n_mfcc (int):               MFCC 的維度
    """
    # 重新取樣到 16k
    if sr != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resample(waveform)
        sr = 16000
    
    # 取出第一個 channel
    signal = waveform[0]
    length = signal.shape[0]
    
    # 計算 MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": frame_size, "hop_length": hop_size, "n_mels": 40}
    )
    
    # 每個 frame 的預測結果
    preds, frame_starts = [], []
    
    for start in range(0, length, hop_size):
        end = start + frame_size
        if end > length:
            break
        
        frame_starts.append(start)
        
        # 取出 frame
        frame = signal[start:end].unsqueeze(0)
        
        mfcc = mfcc_transform(frame).squeeze(0)
        
        # 取平均
        mfcc_mean = mfcc.mean(dim=-1)
        
        # 預測
        with torch.no_grad():
            out = model(mfcc_mean.unsqueeze(0).to(device))
            pred = torch.sign(out).item()
        
        preds.append(pred)
    
    # 合併連續的 speech frame
    segments = []
    in_speech = False
    seg_start = None
    
    for i, pred in enumerate(preds):
        if not in_speech and pred == 1:
            in_speech = True
            seg_start = frame_starts[i]
        elif in_speech and pred == -1:
            in_speech = False
            segments.append((seg_start, frame_starts[i]))
    
    # 強制補齊結束時間
    if in_speech:
        segments.append((seg_start, frame_starts[-1] + frame_size))
    
    return segments, sr

def epd_inference(audio_path):
    if not isinstance(audio_path, str) or not os.path.exists(audio_path):
        return go.Figure()
    
    # 讀取音訊檔案
    audio_path = os.path.abspath(audio_path)
    waveform, sample_rate = torchaudio.load(audio_path)
    
    segments, sr_after = detect_speech_segments(waveform, sample_rate)
    wave_plot = waveform.squeeze().cpu().numpy()
    x_vals = np.arange(len(wave_plot))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=wave_plot, mode='lines', name='waveform'))
    for i, (start_s, end_s) in enumerate(segments):
        fig.add_shape(type="line", x0=start_s, x1=start_s, y0=wave_plot.min(), y1=wave_plot.max(), line=dict(color="red"))
        fig.add_shape(type="line", x0=end_s, x1=end_s, y0=wave_plot.min(), y1=wave_plot.max(), line=dict(color="blue"))
    fig.update_layout(title=f"File: {Path(audio_path).name}", xaxis_title="Sample", yaxis_title="Amplitude")
    return fig

def clear_output():
    return None, go.Figure()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to SVM model")
    args = parser.parse_args()
    
    model = SVM_Model(input_dim=13).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    with gr.Blocks() as demo:
        gr.Markdown("# SVM 語音端點測試 SVM EPD Demo")
        gr.Markdown("##### by D1009212 poyu39")
        output_plot = gr.Plot()
        audio_input = gr.Audio(type="filepath")
        
        with gr.Row():
            submit_button = gr.Button("執行")
            clear_button = gr.Button("清除")
        
        submit_button.click(fn=epd_inference, inputs=audio_input, outputs=output_plot)
        clear_button.click(fn=clear_output, inputs=[], outputs=[audio_input, output_plot])
    
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)
