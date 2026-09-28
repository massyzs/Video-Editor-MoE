"""Cheap complete-output check for resumable inference; no torch dependency."""
import json
from pathlib import Path
import struct
import subprocess
from fractions import Fraction


def _probe_video(video):
    """(width, height, n_frames, fps) of the first video stream: ffprobe if installed, otherwise OpenCV."""
    try:
        data = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate', '-of', 'json', str(video)],
            capture_output=True, text=True, check=True, timeout=30)
    except FileNotFoundError:          # ffprobe not installed
        import cv2
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise OSError(f'cannot open {video}')
            return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), Fraction(cap.get(cv2.CAP_PROP_FPS)).limit_denominator(1000))
        finally:
            cap.release()
    stream = json.loads(data.stdout)['streams'][0]
    return stream['width'], stream['height'], int(stream['nb_frames']), Fraction(stream['avg_frame_rate'])


def output_exists(out_dir, name, frames_png, num_frames=49, width=832, height=480, fps=16):
    video = Path(out_dir) / (name + '.mp4')
    if not video.is_file() or video.stat().st_size == 0:
        return False
    try:
        w, h, n, rate = _probe_video(video)
        if (w, h, n) != (width, height, num_frames):
            return False
        if rate != fps:
            return False
        if frames_png:
            folder = Path(out_dir) / 'frames' / name
            if len(list(folder.glob('f*.png'))) != num_frames:
                return False
            for index in range(num_frames):
                with (folder / f'f{index:03d}.png').open('rb') as handle:
                    header = handle.read(24)
                if (len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n'
                        or header[12:16] != b'IHDR' or struct.unpack('>II', header[16:24]) != (width, height)):
                    return False
        return True
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError, ZeroDivisionError):
        return False
