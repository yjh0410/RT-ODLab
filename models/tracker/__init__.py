from .byte_tracker.build import build_byte_tracker



def build_tracker(args):
    if args.tracker == 'byte_tracker':
        return build_byte_tracker(args)
    else:
        raise NotImplementedError
