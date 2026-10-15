"""Frame Range and Save Audio: the pieces of a chunked upscale."""
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_integration as T  # noqa: E402

nodes = T.nodes


def test_frame_range_takes_a_slice_and_reports_the_total():
    imgs = torch.arange(10).float().view(10, 1, 1, 1).expand(10, 2, 2, 3)
    out = nodes.MiniMaxH3FrameRange().pick(imgs, start=4, length=3)
    part, total = out["result"]
    assert total == 10 and part[:, 0, 0, 0].tolist() == [4, 5, 6]
    assert out["ui"]["frame_range"] == [{"start": 4, "count": 3, "total": 10}]
    last = nodes.MiniMaxH3FrameRange().pick(imgs, start=8, length=5)
    assert last["result"][0].shape[0] == 2                     # fewer at the end
    with pytest.raises(ValueError, match="past the end"):
        nodes.MiniMaxH3FrameRange().pick(imgs, start=10, length=1)
    assert nodes.MiniMaxH3FrameRange.IS_CHANGED() != nodes.MiniMaxH3FrameRange.IS_CHANGED()


def test_frame_range_has_no_4096_limit():
    imgs = torch.zeros(5000, 1, 1, 3)
    part, total = nodes.MiniMaxH3FrameRange().pick(imgs, start=4500, length=400)["result"]
    assert part.shape[0] == 400 and total == 5000


def test_wav_float_writer(tmp_path):
    t = torch.linspace(0, 1, 4800)
    wave = torch.stack([torch.sin(t * 440), torch.cos(t * 440)]) * 0.5
    path = tmp_path / "a.wav"
    nodes.write_wav_float(str(path), wave, 48000)
    raw = path.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    assert int.from_bytes(raw[4:8], "little") == len(raw) - 8
    if shutil.which("ffprobe"):
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                                         capture_output=True, text=True).stdout)["streams"][0]
        assert info["codec_name"] == "pcm_f32le" and info["channels"] == 2 and info["sample_rate"] == "48000"
        assert int(info["duration_ts"]) == 4800


def test_save_audio_stays_in_the_output_folder(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    audio = {"waveform": torch.zeros(1, 2, 480), "sample_rate": 48000}
    ui = nodes.MiniMaxH3SaveAudio().save(audio, "longshot/p/videos/_chunks/Film_audio")["ui"]
    assert ui["longshot_audio"] == [{"filename": "Film_audio.wav", "subfolder": "longshot/p/videos/_chunks",
                                     "type": "output"}]
    assert (tmp_path / "longshot/p/videos/_chunks/Film_audio.wav").is_file()
    with pytest.raises(ValueError):
        nodes.MiniMaxH3SaveAudio().save(audio, "../../escape")


def test_upscale_chunk_streams_to_files(tmp_path, monkeypatch):
    import folder_paths
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    film = torch.rand(30, 16, 24, 3)
    seen = []

    def fake_up(frames):                         # 2x nearest, records batch sizes
        seen.append(int(frames.shape[0]))
        return frames.repeat_interleave(2, 1).repeat_interleave(2, 2)
    out = nodes.MiniMaxH3UpscaleChunk().run(film, start=10, length=15, filename_prefix="longshot/p/videos/_chunks/F_c001",
                                            codec="libx264", master_prefix="longshot/p/videos/_chunks/F_c001_master",
                                            batch=4, _upscale=fake_up, _ffmpeg=shutil.which("ffmpeg"))
    ui = out["ui"]
    assert seen == [4, 4, 4, 3]                       # never more than a batch at a time
    assert ui["frame_range"] == [{"start": 10, "count": 15, "total": 30}]
    v, m = ui["chunk_video"][0], ui["chunk_master"][0]
    assert v == {"filename": "F_c001_00001.mp4", "subfolder": "longshot/p/videos/_chunks", "type": "output"}
    assert m["filename"] == "F_c001_master_00001.mov"
    for e, codec in ((v, "h264"), (m, "prores")):
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json",
                                          str(tmp_path / e["subfolder"] / e["filename"])],
                                         capture_output=True, text=True).stdout)["streams"][0]
        assert info["codec_name"] == codec and int(info["nb_read_frames"]) == 15
        assert (info["width"], info["height"]) == (48, 32)
    assert nodes.MiniMaxH3UpscaleChunk.IS_CHANGED() != nodes.MiniMaxH3UpscaleChunk.IS_CHANGED()


def test_upscale_chunk_at_scale_one_only_encodes_its_frames(tmp_path, monkeypatch):
    """1.6.10: a piece of a film decoded run by run, no RTX upscaler involved."""
    import folder_paths
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(nodes._DirectVSR, "make", staticmethod(lambda *a, **k: pytest.fail("no upscaler at scale 1")))
    film = torch.rand(40, 16, 24, 3)
    out = nodes.MiniMaxH3UpscaleChunk().run(film, start=22, length=10_000_000, scale=1.0,
                                            filename_prefix="longshot/p/videos/_chunks/R_c002",
                                            _ffmpeg=shutil.which("ffmpeg"))
    assert out["ui"]["frame_range"] == [{"start": 22, "count": 18, "total": 40}]
    e = out["ui"]["chunk_video"][0]
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json",
                                      str(tmp_path / e["subfolder"] / e["filename"])],
                                     capture_output=True, text=True).stdout)["streams"][0]
    assert int(info["nb_read_frames"]) == 18 and (info["width"], info["height"]) == (24, 16)


def test_upscale_chunk_drives_the_rtx_upscaler_directly(tmp_path, monkeypatch):
    """One upscaler for the whole chunk, one frame in and out at a time."""
    import types
    import folder_paths
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    log = {"made": 0, "loaded": 0, "frames": [], "size": None}

    class Effect:
        def __init__(self, quality, strength, device, image_encoding):
            log["made"] += 1
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def load(self):
            log["loaded"] += 1
            log["size"] = (self.output_height, self.output_width)
        def run(self, x):                                   # CHW float in, CHW float out (2x nearest)
            log["frames"].append(tuple(x.shape))
            return types.SimpleNamespace(image=x.repeat_interleave(2, 1).repeat_interleave(2, 2))
    nv = types.SimpleNamespace(VideoSuperRes=Effect, QualityLevel={"ULTRA": 4}, ImageEncoding=types.SimpleNamespace(RGB8=0))
    mod = types.SimpleNamespace(nvvfx=nv, get_effect_device=lambda: (torch.device("cpu"), 0))
    direct = nodes._DirectVSR.make(2.0, "ULTRA", 16, 24, module=mod)
    film = torch.rand(30, 16, 24, 3)
    out = nodes.MiniMaxH3UpscaleChunk().run(film, start=5, length=12, filename_prefix="longshot/p/videos/_chunks/D_c000",
                                            codec="libx264", _direct=direct, _ffmpeg=shutil.which("ffmpeg"))
    assert log["made"] == 1 and log["loaded"] == 1 and log["size"] == (32, 48)
    assert log["frames"] == [(3, 16, 24)] * 12              # one frame at a time
    v = out["ui"]["chunk_video"][0]
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json",
                                      str(tmp_path / v["subfolder"] / v["filename"])], capture_output=True, text=True).stdout)["streams"][0]
    assert int(info["nb_read_frames"]) == 12 and (info["width"], info["height"]) == (48, 32)


def test_no_direct_upscaler_without_the_rtx_module():
    import types
    assert nodes._DirectVSR.make(2.0, "ULTRA", 16, 24, module=types.SimpleNamespace()) is None


def test_direct_upscaler_works_with_rtx_vsr_pro(tmp_path, monkeypatch):
    """whmc76's RTX VSR Pro: QualityLevel under nvvfx.effects, VideoSuperRes(quality) only,
    no get_effect_device helper."""
    import types
    import folder_paths
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(nodes, "_cuda_device", lambda: (torch.device("cpu"), 0))
    made = []

    class Effect:
        def __init__(self, quality):
            made.append(quality)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def load(self):
            pass
        def run(self, x):
            return types.SimpleNamespace(image=x.repeat_interleave(2, 1).repeat_interleave(2, 2))
    nv = types.SimpleNamespace(VideoSuperRes=Effect, effects=types.SimpleNamespace(QualityLevel={"ULTRA": "U"}))
    direct = nodes._DirectVSR.make(2.0, "ULTRA", 16, 24, module=types.SimpleNamespace(nvvfx=nv))
    assert direct is not None
    out = nodes.MiniMaxH3UpscaleChunk().run(torch.rand(20, 16, 24, 3), start=0, length=20,
                                            filename_prefix="longshot/p/videos/_chunks/P_c000", codec="libx264",
                                            _direct=direct, _ffmpeg=shutil.which("ffmpeg"))
    assert made == ["U"]
    v = out["ui"]["chunk_video"][0]
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json",
                                      str(tmp_path / v["subfolder"] / v["filename"])], capture_output=True, text=True).stdout)["streams"][0]
    assert int(info["nb_read_frames"]) == 20


def test_huge_outputs_go_through_the_node():
    import types
    nv = types.SimpleNamespace(VideoSuperRes=object, QualityLevel={"ULTRA": 1})
    mod = types.SimpleNamespace(nvvfx=nv, get_effect_device=lambda: (torch.device("cpu"), 0))
    assert nodes._DirectVSR.make(4.0, "ULTRA", 4000, 4000, module=mod) is None      # 16000 px edge


def test_a_dead_encoder_is_reported(tmp_path, monkeypatch):
    import folder_paths
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setitem(nodes._CHUNK_CODECS, "libx264", ["-c:v", "no_such_encoder"])
    with pytest.raises(RuntimeError):
        nodes.MiniMaxH3UpscaleChunk().run(torch.rand(40, 16, 24, 3), start=0, length=40,
                                          filename_prefix="longshot/p/videos/_chunks/X_c000", codec="libx264",
                                          _upscale=lambda f: f, _ffmpeg=shutil.which("ffmpeg"))


def test_save_video_and_load_video_round_trip(tmp_path, monkeypatch):
    """1.6.11: video files without Video Helper Suite."""
    import folder_paths
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "out"))
    film = torch.rand(30, 32, 48, 3)
    audio = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}
    ui = nodes.MiniMaxH3SaveVideo().save(film, "longshot/p/videos/Film", "libx264", 24.0, audio=audio, _ffmpeg=ff)["ui"]
    e = ui["gifs"][0]
    assert e["filename"] == "Film_00001-audio.mp4" and e["subfolder"] == "longshot/p/videos"
    again = nodes.MiniMaxH3SaveVideo().save(film, "longshot/p/videos/Film", "prores_hq", 24.0, _ffmpeg=ff)["ui"]["gifs"][0]
    assert again["filename"] == "Film_00002.mov"
    # load it back as a clip: 24 fps, frames 6.., scaled
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda name, *a: e["fullpath"])
    imgs, count, a = nodes.MiniMaxH3LoadVideo().load("clip.mp4", skip_first_frames=6, frame_load_cap=12,
                                                     width=24, height=16, _ffmpeg=ff)
    assert count == 12 and tuple(imgs.shape) == (12, 16, 24, 3)
    assert a["sample_rate"] == 48000 and a["waveform"].shape[-1] == 24000


@pytest.mark.parametrize("with_audio", [True, False])
def test_save_video_carries_a_workflow_longer_than_a_windows_command_line(tmp_path, monkeypatch, with_audio):
    """1.6.16: WinError 206. The workflow goes in an ffmetadata file, not on the command line."""
    import folder_paths
    import json
    import subprocess
    ff, probe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ff or not probe:
        pytest.skip("no ffmpeg / ffprobe")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "out"))
    seen = []
    real = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, *a, **k: (seen.append(cmd), real(cmd, *a, **k))[1])
    tricky = 'a=b; c#d \\ "quoted"\nnew line ' * 2000                 # > 32767 characters
    workflow = {"nodes": [{"id": i, "text": tricky[:100] + str(i)} for i in range(400)], "note": tricky}
    prompt = {"1": {"class_type": "X", "inputs": {"text": tricky}}}
    audio = {"waveform": torch.zeros(1, 2, 24000), "sample_rate": 48000} if with_audio else None
    e = nodes.MiniMaxH3SaveVideo().save(torch.rand(12, 32, 48, 3), "longshot/p/videos/Big", "libx264", 24.0,
                                        audio=audio, prompt=prompt,
                                        extra_pnginfo={"workflow": workflow}, _ffmpeg=ff)["ui"]["gifs"][0]
    assert sum(len(x) for x in seen[-1]) < 8000                       # the command line stays short
    out = subprocess.run([probe, "-v", "error", "-show_entries", "format_tags=workflow,prompt", "-of", "json",
                          e["fullpath"]], capture_output=True, text=True, encoding="utf-8")
    tags = {k.lower(): v for k, v in json.loads(out.stdout)["format"]["tags"].items()}
    assert json.loads(tags["workflow"]) == workflow and json.loads(tags["prompt"]) == prompt
    assert sorted(os.listdir(os.path.dirname(e["fullpath"]))) == [e["filename"]]   # no temp files left
