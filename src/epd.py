import argparse
import gradio as gr
import torch
import torchaudio
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path
from model import SVM_Model
from tqdm import tqdm
import csv

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
    Returns:
        segments (List[Tuple[int, int]]): speech frame 的起始與結束時間
        sr (int):                         重新取樣後的取樣率
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
    parser.add_argument("--model", type=str, required=True, help="SVM 模型路徑")
    parser.add_argument("--dataset", type=str, help="測試資料集路徑")
    parser.add_argument("--score", type=bool, default=False, help="是否計算 EPD Score")
    args = parser.parse_args()
    
    model = SVM_Model(input_dim=13).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    if args.score:
        epd_score_csv = Path(args.model).parent / "epd_score.csv"
        
        csv_header = ["Speaker", "Correct", "Total", "EPD Score"]
        csv_file = open(epd_score_csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)
        csv_file.close()
        
        directory_path = Path(args.dataset)
        
        correct = 0
        total = 0
        
        speakers = [d for d in directory_path.iterdir() if d.is_dir()]
        wav_files = [f for s in speakers for f in s.glob("*.wav")]
        
        epd_scores = []
        for speaker in speakers:
            wav_files = list(speaker.glob("*.wav"))
            pbar = tqdm(wav_files, desc=f"計算語者 {speaker.name} 分數")
            pbar.set_description(f"計算語者 {speaker.name} 分數")
            for file_path in pbar:
                _, start_str, end_str = file_path.stem.split('_')
                start_sample = int(start_str)
                end_sample   = int(end_str)
                
                audio_path = os.path.abspath(file_path)
                waveform, sample_rate = torchaudio.load(audio_path)
                segments, sr_after = detect_speech_segments(waveform, sample_rate)
                
                # 強制只取第一個 segment 起始 frame 與最後一個 segment 結束 frame
                if len(segments) == 0:
                    continue
                pred_start_sample = segments[0][0]
                pred_end_sample = segments[-1][1]
                
                # 容忍誤差 1000 個 sample
                te = 1000
                
                if abs(pred_start_sample - start_sample) <= te and abs(pred_end_sample - end_sample) <= te:
                    correct += 1
                elif abs(pred_start_sample - start_sample) <= te or abs(pred_end_sample - end_sample) <= te:
                    correct += 0.5
                total += 1
                
                pbar.set_postfix({"EPD Score": f"{correct}/{total} ({correct / total * 100:.2f}%)"})
            
            csv_file = open(epd_score_csv, "a", newline="")
            writer = csv.writer(csv_file)
            writer.writerow([speaker.name, correct, total, correct / total * 100])
            csv_file.close()
            epd_scores.append(correct / total * 100)
        csv_file = open(epd_score_csv, "a", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["Average", 0, 0, sum(epd_scores) / len(epd_scores)])
        csv_file.close()
    else:
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
