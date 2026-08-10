"""Detection pipeline shutdown helpers (seek / stop)."""

import queue
import threading

from app.processors.video_processor import VideoProcessor


def _bare_processor() -> VideoProcessor:
    processor = VideoProcessor.__new__(VideoProcessor)
    processor._raw_frame_queue = queue.Queue(maxsize=8)
    processor._detection_pipeline_thread = None
    processor.worker_threads = []
    return processor


def test_drain_raw_frame_queue_drops_pending_tasks():
    processor = _bare_processor()
    rq = processor._raw_frame_queue
    rq.put((0, 0, None, {}, {}, None, False, "video"))
    rq.put((1, 1, None, {}, {}, None, False, "video"))

    dropped = processor._drain_raw_frame_queue()

    assert dropped == 2
    assert rq.empty()


def test_prepare_detection_pipeline_join_drains_and_signals_end():
    processor = _bare_processor()
    rq = processor._raw_frame_queue
    rq.put((0, 0, None, {}, {}, None, False, "video"))

    keep_alive = threading.Event()

    def _block() -> None:
        keep_alive.wait(timeout=2.0)

    t = threading.Thread(target=_block, daemon=True)
    t.start()
    processor._detection_pipeline_thread = t

    dropped = processor._prepare_detection_pipeline_join()

    assert dropped == 1
    assert rq.get_nowait() is None
    keep_alive.set()
    t.join(timeout=2.0)


def test_pool_workers_can_be_cancelled_before_shutdown_joins():
    processor = _bare_processor()
    first = type("_Worker", (), {"stop_event": threading.Event(), "name": "first"})()
    second = type("_Worker", (), {"stop_event": threading.Event(), "name": "second"})()
    processor.worker_threads = [first, second]

    processor._signal_pool_workers_to_stop()

    assert first.stop_event.is_set()
    assert second.stop_event.is_set()
