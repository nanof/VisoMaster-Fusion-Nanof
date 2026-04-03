
from pathlib import Path
p = Path('app/processors/workers/frame_worker.py')
t = p.read_text(encoding='utf-8')
old = '''        return {
            "feeder": dict(fd) if fd else {},
            "worker": stages,
            "worker_total_ms": total,
            "worker_thread": self.name,
        }

    # ----------------------------------------------------------------
    # VR1-80 helpers'''
new = '''        fd_dict = dict(fd) if fd else {}
        feeder_total = 0.0
        for _k, _v in fd_dict.items():
            try:
                feeder_total += float(_v)
            except (TypeError, ValueError):
                pass
        q_depth, q_max = 0, 0
        fq = getattr(self.video_processor, "frame_queue", None)
        if eq is not None:
            try:
                q_depth = fq.qsize()
                q_max = int(fq.maxsize)
            except (TypeError, ValueError, AttributeError):
                pass
        return {
            "feeder": fd_dict,
            "worker": stages,
            "worker_total_ms": total,
            "feeder_total_ms": feeder_total,
            "worker_thread": self.name,
            "frame_number": int(self.frame_number),
            "frame_queue_depth_at_emit": q_depth,
            "frame_queue_max": q_max,
        }

    # ----------------------------------------------------------------
    # VR180 helpers'''
assert old in t, 'missing'
p.write_text(t.replace(old, new, 1), encoding='utf-8')
print('ok')
