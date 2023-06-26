from .image_encoder.img_encoder import build_img_encoder


# build encoder
def build_encoder(cfg, trainable=False, en_type='img_encoder'):
    if en_type == 'img_encoder':
        return build_img_encoder(cfg, trainable)
    elif en_type == 'text_encoder':
        ## TODO: design text encoder
        return None
