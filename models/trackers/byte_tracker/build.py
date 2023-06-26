from .byte_tracker import ByteTracker


def build_byte_tracker(args):
    tracker = ByteTracker(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        frame_rate=args.fps,
        match_thresh=args.match_thresh,
        mot20=args.mot20
    )

    return tracker
    